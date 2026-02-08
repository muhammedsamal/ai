package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"github.com/atotto/clipboard"
	"github.com/joho/godotenv"
	"io"
	"math/rand"
	"net/http"
	"os"
	"strings"
	"time"
)

type Response struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
	Model string `json:"model"`
}

type ResponsesAPIResponse struct {
	Output []struct {
		Content []struct {
			Text string `json:"text"`
		} `json:"content"`
	} `json:"output"`
	Usage struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
	Model string `json:"model"`
}

type CloudflareResponse struct {
	Result struct {
		Response string `json:"response"`
	} `json:"result"`
	Success bool `json:"success"`
}

type AIResult struct {
	Content  string
	Model    string
	Tokens   int
	Duration time.Duration
}

const SYSTEM_PROMPT = `Be extremely concise. Sacrifice grammar for the sake of concision. Respond in plain text only. No markdown, no bold, no asterisks.`

func main() {
	godotenv.Load()
	// join all args after program name
	if len(os.Args) < 2 {
		fmt.Println("Usage: at {prompt}")
		return
	}

	prompt := strings.Join(os.Args[1:], " ")

	// Spinner start
	done := make(chan bool)
	go func() {
		chars := []string{"|", "/", "-", "\\"}
		i := 0
		for {
			select {
			case <-done:
				// overwrite the line with spaces, then return to the start
				fmt.Print("\r          \r")
				return
			default:
				fmt.Printf("\rThinking %s", chars[i%len(chars)])
				i++
				time.Sleep(100 * time.Millisecond)
			}
		}
	}()

	result, err := ai(prompt)
	done <- true // stop spinner
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	content := strings.TrimSpace(result.Content)
	fmt.Println()
	fmt.Println(content)
	fmt.Printf("\n[%s | %d tokens | %s]\n", result.Model, result.Tokens, result.Duration.Round(time.Millisecond))

	// copy to clipboard
	clipboard.WriteAll(content)

}

func ai(input string) (AIResult, error) {
	providers := []func(string) (AIResult, error){groqAPI, vercelAIGateway, openAIResponses, deepseekAPI, cloudflareAI}
	return providers[rand.Intn(len(providers))](input)
}

func groqAPI(input string) (AIResult, error) {
	return callAPI("GROQ_API_KEY", "https://api.groq.com/openai/v1/chat/completions", "moonshotai/kimi-k2-instruct-0905", input)
}

func deepseekAPI(input string) (AIResult, error) {
	return callAPI("DEEPSEEK_API_KEY", "https://api.deepseek.com/chat/completions", "deepseek-chat", input)
}

func vercelAIGateway(input string) (AIResult, error) {
	return callAPI("AI_GATEWAY_API_KEY", "https://ai-gateway.vercel.sh/v1/chat/completions", "google/gemini-3-flash", input)
}

func callAPI(keyEnv, url, model, input string) (AIResult, error) {
	apiKey := os.Getenv(keyEnv)
	if apiKey == "" {
		return AIResult{}, fmt.Errorf("%s is not set", keyEnv)
	}

	payload := map[string]interface{}{
		"model":  model,
		"stream": false,
		"messages": []map[string]interface{}{
			{"role": "system", "content": SYSTEM_PROMPT},
			{"role": "user", "content": input},
		},
	}

	jsonBody, _ := json.Marshal(payload)
	req, _ := http.NewRequest("POST", url, bytes.NewBuffer(jsonBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)

	start := time.Now()
	resp, err := (&http.Client{}).Do(req)
	duration := time.Since(start)
	if err != nil {
		return AIResult{}, err
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)

	var res Response
	if err := json.Unmarshal(body, &res); err != nil {
		return AIResult{}, err
	}

	result := AIResult{
		Model:    res.Model,
		Tokens:   res.Usage.TotalTokens,
		Duration: duration,
	}

	if len(res.Choices) > 0 {
		result.Content = res.Choices[0].Message.Content
		return result, nil
	}

	result.Content = "No response content"
	return result, nil
}

func openAIResponses(input string) (AIResult, error) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		return AIResult{}, fmt.Errorf("OPENAI_API_KEY is not set")
	}

	payload := map[string]interface{}{
		"model":        "gpt-5.2",
		"instructions": SYSTEM_PROMPT,
		"input":        input,
	}

	jsonBody, _ := json.Marshal(payload)
	req, _ := http.NewRequest("POST", "https://api.openai.com/v1/responses", bytes.NewBuffer(jsonBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)

	start := time.Now()
	resp, err := (&http.Client{}).Do(req)
	duration := time.Since(start)
	if err != nil {
		return AIResult{}, err
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)

	var res ResponsesAPIResponse
	if err := json.Unmarshal(body, &res); err != nil {
		return AIResult{}, err
	}

	result := AIResult{
		Model:    res.Model,
		Tokens:   res.Usage.TotalTokens,
		Duration: duration,
	}

	if len(res.Output) > 0 && len(res.Output[0].Content) > 0 {
		result.Content = res.Output[0].Content[0].Text
		return result, nil
	}

	result.Content = "No response content"
	return result, nil
}

func cloudflareAI(input string) (AIResult, error) {
	token := os.Getenv("CLOUDFLARE_API_TOKEN")
	accountID := os.Getenv("CLOUDFLARE_ACCOUNT_ID")
	if token == "" || accountID == "" {
		return AIResult{}, fmt.Errorf("CLOUDFLARE_API_TOKEN or CLOUDFLARE_ACCOUNT_ID is not set")
	}

	model := "@cf/openai/gpt-oss-120b"
	url := fmt.Sprintf("https://api.cloudflare.com/client/v4/accounts/%s/ai/run/%s", accountID, model)

	payload := map[string]interface{}{
		"prompt": SYSTEM_PROMPT + "\n\n" + input,
	}

	jsonBody, _ := json.Marshal(payload)
	req, _ := http.NewRequest("POST", url, bytes.NewBuffer(jsonBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+token)

	start := time.Now()
	resp, err := (&http.Client{}).Do(req)
	duration := time.Since(start)
	if err != nil {
		return AIResult{}, err
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)

	var res CloudflareResponse
	if err := json.Unmarshal(body, &res); err != nil {
		return AIResult{}, err
	}

	if !res.Success {
		return AIResult{}, fmt.Errorf("cloudflare API returned unsuccessful response")
	}

	return AIResult{
		Content:  res.Result.Response,
		Model:    model,
		Duration: duration,
	}, nil
}

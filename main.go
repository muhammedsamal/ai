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
	"sync"
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

type GeminiResponse struct {
	Candidates []struct {
		Content struct {
			Parts []struct {
				Text string `json:"text"`
			} `json:"parts"`
		} `json:"content"`
	} `json:"candidates"`
	UsageMetadata struct {
		PromptTokenCount     int `json:"promptTokenCount"`
		CandidatesTokenCount int `json:"candidatesTokenCount"`
		TotalTokenCount      int `json:"totalTokenCount"`
	} `json:"usageMetadata"`
	ModelVersion string `json:"modelVersion"`
}

type CloudflareResponse struct {
	Result struct {
		Response string `json:"response"`
		Choices  []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Usage struct {
			TotalTokens int `json:"total_tokens"`
		} `json:"usage"`
		Model string `json:"model"`
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

	args := os.Args[1:]
	if len(args) < 1 {
		fmt.Println("Usage: ai [-a] {prompt}")
		return
	}

	allMode := false
	if args[0] == "-a" {
		allMode = true
		args = args[1:]
	}

	if len(args) < 1 {
		fmt.Println("Usage: ai [-a] {prompt}")
		return
	}

	prompt := strings.Join(args, " ")

	if allMode {
		aiAll(prompt)
	} else {
		stop := spinner()
		result, err := ai(prompt)
		stop()
		if err != nil {
			fmt.Println("Error:", err)
			return
		}

		content := strings.TrimSpace(result.Content)
		fmt.Println()
		fmt.Println(content)
		fmt.Printf("\n[%s | %d tokens | %s]\n", result.Model, result.Tokens, result.Duration.Round(time.Millisecond))
		clipboard.WriteAll(content)
	}
}

func spinner() func() {
	done := make(chan bool)
	go func() {
		chars := []string{"|", "/", "-", "\\"}
		i := 0
		for {
			select {
			case <-done:
				fmt.Print("\r          \r")
				return
			default:
				fmt.Printf("\rThinking %s", chars[i%len(chars)])
				i++
				time.Sleep(100 * time.Millisecond)
			}
		}
	}()
	return func() { done <- true }
}

func aiAll(input string) {
	providers := []func(string) (AIResult, error){groqAPI, vercelAIGateway, openAIResponses, deepseekAPI, cloudflareAI, geminiAPI, openRouterAPI}
	results := make(chan AIResult, len(providers))
	var wg sync.WaitGroup

	for _, p := range providers {
		wg.Add(1)
		go func(fn func(string) (AIResult, error)) {
			defer wg.Done()
			result, err := fn(input)
			if err == nil {
				results <- result
			}
		}(p)
	}

	go func() {
		wg.Wait()
		close(results)
	}()

	for result := range results {
		content := strings.TrimSpace(result.Content)
		fmt.Printf("\n--- %s ---\n%s\n[%d tokens | %s]\n", result.Model, content, result.Tokens, result.Duration.Round(time.Millisecond))
	}
}

func ai(input string) (AIResult, error) {
	providers := []func(string) (AIResult, error){groqAPI, vercelAIGateway, openAIResponses, deepseekAPI, cloudflareAI, geminiAPI, openRouterAPI}
	rand.Shuffle(len(providers), func(i, j int) { providers[i], providers[j] = providers[j], providers[i] })
	var lastErr error
	for _, p := range providers {
		result, err := p(input)
		if err == nil {
			return result, nil
		}
		lastErr = err
	}
	return AIResult{}, fmt.Errorf("all providers failed: %w", lastErr)
}

func groqAPI(input string) (AIResult, error) {
	return callAPI("GROQ_API_KEY", "https://api.groq.com/openai/v1/chat/completions", "moonshotai/kimi-k2-instruct-0905", input)
}

func deepseekAPI(input string) (AIResult, error) {
	return callAPI("DEEPSEEK_API_KEY", "https://api.deepseek.com/chat/completions", "deepseek-chat", input)
}

func openRouterAPI(input string) (AIResult, error) {
	return callAPI("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1/chat/completions", "openrouter/aurora-alpha", input)
}

func vercelAIGateway(input string) (AIResult, error) {
	return callAPI("AI_GATEWAY_API_KEY", "https://ai-gateway.vercel.sh/v1/chat/completions", "anthropic/claude-haiku-4.5", input)
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
		"model":        "gpt-5-mini-2025-08-07",
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

	for _, output := range res.Output {
		for _, c := range output.Content {
			if c.Text != "" {
				result.Content = c.Text
				return result, nil
			}
		}
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
		"messages": []map[string]string{
			{"role": "system", "content": SYSTEM_PROMPT},
			{"role": "user", "content": input},
		},
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

	result := AIResult{
		Model:    model,
		Tokens:   res.Result.Usage.TotalTokens,
		Duration: duration,
	}

	if len(res.Result.Choices) > 0 {
		result.Content = res.Result.Choices[0].Message.Content
	} else if res.Result.Response != "" {
		result.Content = res.Result.Response
	} else {
		result.Content = "No response content"
	}

	return result, nil
}

func geminiAPI(input string) (AIResult, error) {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		return AIResult{}, fmt.Errorf("GEMINI_API_KEY is not set")
	}

	model := "gemini-3-flash-preview"
	url := fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent", model)

	payload := map[string]interface{}{
		"system_instruction": map[string]interface{}{
			"parts": []map[string]string{{"text": SYSTEM_PROMPT}},
		},
		"contents": []map[string]interface{}{
			{"parts": []map[string]string{{"text": input}}},
		},
	}

	jsonBody, _ := json.Marshal(payload)
	req, _ := http.NewRequest("POST", url, bytes.NewBuffer(jsonBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-goog-api-key", apiKey)

	start := time.Now()
	resp, err := (&http.Client{}).Do(req)
	duration := time.Since(start)
	if err != nil {
		return AIResult{}, err
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)

	var res GeminiResponse
	if err := json.Unmarshal(body, &res); err != nil {
		return AIResult{}, err
	}

	result := AIResult{
		Model:    res.ModelVersion,
		Tokens:   res.UsageMetadata.TotalTokenCount,
		Duration: duration,
	}

	if len(res.Candidates) > 0 && len(res.Candidates[0].Content.Parts) > 0 {
		result.Content = res.Candidates[0].Content.Parts[0].Text
		return result, nil
	}

	result.Content = "No response content"
	return result, nil
}

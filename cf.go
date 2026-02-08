//go:build ignore

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"

	"github.com/joho/godotenv"
)

type CFResponse struct {
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

func main() {
	godotenv.Load()

	token := os.Getenv("CLOUDFLARE_API_TOKEN")
	accountID := os.Getenv("CLOUDFLARE_ACCOUNT_ID")
	if token == "" || accountID == "" {
		fmt.Println("CLOUDFLARE_API_TOKEN or CLOUDFLARE_ACCOUNT_ID is not set")
		return
	}

	model := "@cf/openai/gpt-oss-120b"
	url := fmt.Sprintf("https://api.cloudflare.com/client/v4/accounts/%s/ai/run/%s", accountID, model)

	payload := map[string]interface{}{
		"messages": []map[string]string{
			{"role": "system", "content": "Be concise."},
			{"role": "user", "content": "What is 2+2?"},
		},
	}

	jsonBody, _ := json.Marshal(payload)
	fmt.Println("Request URL:", url)
	fmt.Println("Request Body:", string(jsonBody))
	fmt.Println()

	req, _ := http.NewRequest("POST", url, bytes.NewBuffer(jsonBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+token)

	resp, err := (&http.Client{}).Do(req)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	defer resp.Body.Close()

	fmt.Println("Status:", resp.StatusCode)
	fmt.Println()

	body, _ := io.ReadAll(resp.Body)
	fmt.Println("Raw Response:")
	fmt.Println(string(body))
	fmt.Println()

	var res CFResponse
	if err := json.Unmarshal(body, &res); err != nil {
		fmt.Println("Unmarshal error:", err)
		return
	}

	fmt.Println("Parsed:")
	fmt.Println("  Success:", res.Success)
	fmt.Println("  Tokens:", res.Result.Usage.TotalTokens)
	if len(res.Result.Choices) > 0 {
		fmt.Println("  Content (choices):", res.Result.Choices[0].Message.Content)
	}
	if res.Result.Response != "" {
		fmt.Println("  Content (response):", res.Result.Response)
	}
}

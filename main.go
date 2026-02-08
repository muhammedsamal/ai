package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"github.com/atotto/clipboard"
	"io"
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
}

const SYSTEM_PROMPT = `Be extremely concise. Sacrifice grammar for the sake of concision. Respond in plain text only. No markdown, no bold, no asterisks.`

func main() {
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

	res, err := ai(prompt)
	done <- true // stop spinner
	res = strings.TrimSpace(res)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	fmt.Println()
	fmt.Println(res)
	fmt.Println()

	// copy to clipboard
	clipboard.WriteAll(res)

}

func ai(input string) (string, error) {
	apiKey := os.Getenv("GROQ_API_KEY")
	if apiKey == "" {
		return "", fmt.Errorf("GROQ_API_KEY is not set")
	}
	url := "https://api.groq.com/openai/v1/chat/completions"

	payload := map[string]interface{}{
		"model": "moonshotai/kimi-k2-instruct-0905",
		"messages": []map[string]interface{}{
			{
				"role":    "system",
				"content": SYSTEM_PROMPT,
			},
			{
				"role":    "user",
				"content": input,
			},
		},
	}

	jsonBody, _ := json.Marshal(payload)
	req, _ := http.NewRequest("POST", url, bytes.NewBuffer(jsonBody))

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)

	var res Response
	if err := json.Unmarshal(body, &res); err != nil {
		return "", err
	}

	if len(res.Choices) > 0 {
		return res.Choices[0].Message.Content, nil
	}

	return "No response content", nil
}

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/vllm-intelligent-scheduler/internal/api"
	"github.com/vllm-intelligent-scheduler/internal/config"
)

var (
	requestsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{Name: "gateway_requests_total", Help: "Total API requests"},
		[]string{"status"},
	)
	queueDepth = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "gateway_queue_depth", Help: "Current scheduler queue depth",
	})
)

func init() {
	prometheus.MustRegister(requestsTotal, queueDepth)
}

func main() {
	cfgPath := os.Getenv("CONFIG_PATH")
	if cfgPath == "" {
		cfgPath = "configs/gateway.yaml"
	}
	cfg, err := config.Load(cfgPath)
	if err != nil {
		log.Fatalf("config: %v", err)
	}

	engine := api.NewEngine()
	gw := &Gateway{cfg: cfg, engine: engine, client: &http.Client{Timeout: 300 * time.Second}}

	for i := 0; i < cfg.Runtime.MaxWorkers; i++ {
		go gw.dispatchLoop()
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"status":"ok"}`))
	})
	mux.HandleFunc("/ready", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	mux.Handle("/metrics", promhttp.Handler())
	mux.HandleFunc("/scheduler/stats", func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(gw.engine.Stats())
	})
	mux.HandleFunc("/v1/chat/completions", gw.handleChatCompletions)

	addr := fmt.Sprintf("%s:%d", cfg.Server.Host, cfg.Server.Port)
	log.Printf("gateway listening on %s -> vllm %s", addr, cfg.VLLM.BaseURL)
	log.Fatal(http.ListenAndServe(addr, mux))
}

type Gateway struct {
	cfg    *config.Config
	engine *api.Engine
	client *http.Client
}

type chatRequest struct {
	Model    string                   `json:"model"`
	Messages []map[string]interface{} `json:"messages"`
	MaxTokens int                     `json:"max_tokens"`
	Stream   bool                     `json:"stream"`
}

func (g *Gateway) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	var req chatRequest
	if err := json.Unmarshal(body, &req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	prompt := extractPrompt(req.Messages)
	predicted := predictLength(prompt, req.MaxTokens)

	sr := &api.Request{
		ID:              fmt.Sprintf("%d", time.Now().UnixNano()),
		Prompt:          prompt,
		MaxTokens:       req.MaxTokens,
		PredictedTokens: predicted,
		Result:          make(chan map[string]interface{}, 1),
	}
	g.engine.Submit(sr)
	queueDepth.Set(float64(g.engine.Depth()))

	if req.Stream {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "streaming unsupported", http.StatusInternalServerError)
			return
		}
		resp, err := g.forwardStream(body)
		if err != nil {
			requestsTotal.WithLabelValues("error").Inc()
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		defer resp.Body.Close()
		buf := make([]byte, 4096)
		for {
			n, err := resp.Body.Read(buf)
			if n > 0 {
				_, _ = w.Write(buf[:n])
				flusher.Flush()
			}
			if err != nil {
				break
			}
		}
		g.engine.Complete(sr.ID, true)
		requestsTotal.WithLabelValues("success").Inc()
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 300*time.Second)
	defer cancel()
	acquired, err := g.engine.Acquire(ctx)
	if err != nil {
		requestsTotal.WithLabelValues("timeout").Inc()
		http.Error(w, "scheduler timeout", http.StatusGatewayTimeout)
		return
	}
	_ = acquired

	resp, err := g.forward(body)
	if err != nil {
		g.engine.Complete(sr.ID, false)
		requestsTotal.WithLabelValues("error").Inc()
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	g.engine.Complete(sr.ID, true)
	requestsTotal.WithLabelValues("success").Inc()
	w.Header().Set("Content-Type", "application/json")
	w.Write(resp)
}

func (g *Gateway) forward(body []byte) ([]byte, error) {
	url := g.cfg.VLLM.BaseURL + "/v1/chat/completions"
	req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := g.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	return io.ReadAll(resp.Body)
}

func (g *Gateway) forwardStream(body []byte) (*http.Response, error) {
	url := g.cfg.VLLM.BaseURL + "/v1/chat/completions"
	req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	return g.client.Do(req)
}

func (g *Gateway) dispatchLoop() {
	for {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		_, err := g.engine.Acquire(ctx)
		cancel()
		if err != nil {
			continue
		}
	}
}

func extractPrompt(messages []map[string]interface{}) string {
	var buf bytes.Buffer
	for _, m := range messages {
		role, _ := m["role"].(string)
		content, _ := m["content"].(string)
		fmt.Fprintf(&buf, "%s: %s\n", role, content)
	}
	return buf.String()
}

func predictLength(prompt string, maxTokens int) float64 {
	base := float64(len(prompt)) * 0.15
	if base < 10 {
		base = 10
	}
	if base > float64(maxTokens) {
		return float64(maxTokens)
	}
	return base
}

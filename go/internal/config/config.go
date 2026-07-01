package config

import (
	"os"

	"gopkg.in/yaml.v3"
)

type Config struct {
	Server   ServerConfig   `yaml:"server"`
	VLLM     VLLMConfig     `yaml:"vllm"`
	Runtime  RuntimeConfig  `yaml:"runtime"`
	Auth     AuthConfig     `yaml:"auth"`
	Metrics  MetricsConfig  `yaml:"metrics"`
}

type ServerConfig struct {
	Host         string `yaml:"host"`
	Port         int    `yaml:"port"`
	ReadTimeout  int    `yaml:"read_timeout_sec"`
	WriteTimeout int    `yaml:"write_timeout_sec"`
}

type VLLMConfig struct {
	BaseURL string `yaml:"base_url"`
	Model   string `yaml:"model"`
}

type RuntimeConfig struct {
	SchedulerType string `yaml:"scheduler_type"`
	MaxWorkers    int    `yaml:"max_workers"`
	ModelPath     string `yaml:"model_path"`
}

type AuthConfig struct {
	Enabled   bool   `yaml:"enabled"`
	JWTSecret string `yaml:"jwt_secret"`
}

type MetricsConfig struct {
	Enabled bool `yaml:"enabled"`
	Port    int  `yaml:"port"`
}

func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Default(), nil
	}
	cfg := Default()
	if err := yaml.Unmarshal(data, cfg); err != nil {
		return nil, err
	}
	return cfg, nil
}

func Default() *Config {
	return &Config{
		Server: ServerConfig{Host: "0.0.0.0", Port: 8080, ReadTimeout: 300, WriteTimeout: 300},
		VLLM:   VLLMConfig{BaseURL: "http://localhost:8002", Model: "HuggingFaceTB/SmolLM3-3B"},
		Runtime: RuntimeConfig{SchedulerType: "sjf", MaxWorkers: 32, ModelPath: "shared/models/output_length.onnx"},
		Auth:   AuthConfig{Enabled: false},
		Metrics: MetricsConfig{Enabled: true, Port: 9091},
	}
}

package api

import (
	"context"
	"testing"
	"time"
)

func TestEngineSJFOrdering(t *testing.T) {
	e := NewEngine()
	short := &Request{ID: "short", PredictedTokens: 10, MaxTokens: 10}
	long := &Request{ID: "long", PredictedTokens: 100, MaxTokens: 100}
	e.Submit(long)
	e.Submit(short)

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	first, err := e.Acquire(ctx)
	if err != nil {
		t.Fatalf("acquire: %v", err)
	}
	if first.ID != "short" {
		t.Fatalf("expected short first, got %s", first.ID)
	}
}

func TestEngineCancel(t *testing.T) {
	e := NewEngine()
	req := &Request{ID: "x", PredictedTokens: 5, MaxTokens: 5}
	e.Submit(req)
	if !e.Cancel("x") {
		t.Fatal("expected cancel to succeed")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	_, err := e.Acquire(ctx)
	if err == nil {
		t.Fatal("expected timeout after cancel")
	}
}

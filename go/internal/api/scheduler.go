package api

import (
	"container/heap"
	"context"
	"sync"
	"time"
)

type Request struct {
	ID               string
	Prompt           string
	MaxTokens        int
	PredictedTokens  float64
	Priority         float64
	EnqueuedAt       time.Time
	Result           chan map[string]interface{}
	Stream           chan []byte
}

type item struct {
	priority   float64
	enqueuedAt time.Time
	req        *Request
	index      int
}

type priorityHeap []*item

func (h priorityHeap) Len() int           { return len(h) }
func (h priorityHeap) Less(i, j int) bool { return h[i].priority < h[j].priority }
func (h priorityHeap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
	h[i].index = i
	h[j].index = j
}
func (h *priorityHeap) Push(x interface{}) {
	it := x.(*item)
	it.index = len(*h)
	*h = append(*h, it)
}
func (h *priorityHeap) Pop() interface{} {
	old := *h
	n := len(old)
	it := old[n-1]
	old[n-1] = nil
	it.index = -1
	*h = old[:n-1]
	return it
}

type Engine struct {
	mu       sync.Mutex
	cond     *sync.Cond
	heap     priorityHeap
	pending  map[string]*Request
	cancelled map[string]struct{}
	scheduled uint64
	completed uint64
	agingFactor float64
}

func NewEngine() *Engine {
	e := &Engine{
		pending:   make(map[string]*Request),
		cancelled: make(map[string]struct{}),
		agingFactor: 0.001,
	}
	e.cond = sync.NewCond(&e.mu)
	heap.Init(&e.heap)
	go e.agingLoop()
	return e
}

func (e *Engine) Submit(req *Request) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if req.PredictedTokens <= 0 {
		req.PredictedTokens = float64(req.MaxTokens)
	}
	req.Priority = req.PredictedTokens
	req.EnqueuedAt = time.Now()
	e.pending[req.ID] = req
	heap.Push(&e.heap, &item{priority: req.Priority, enqueuedAt: req.EnqueuedAt, req: req})
	e.scheduled++
	e.cond.Signal()
}

func (e *Engine) Acquire(ctx context.Context) (*Request, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	for {
		for e.heap.Len() > 0 {
			it := heap.Pop(&e.heap).(*item)
			if _, ok := e.cancelled[it.req.ID]; ok {
				delete(e.cancelled, it.req.ID)
				delete(e.pending, it.req.ID)
				continue
			}
			delete(e.pending, it.req.ID)
			return it.req, nil
		}
		done := make(chan struct{})
		go func() {
			select {
			case <-ctx.Done():
				e.cond.Broadcast()
			case <-done:
			}
		}()
		e.cond.Wait()
		close(done)
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
	}
}

func (e *Engine) Complete(id string, success bool) {
	if success {
		e.completed++
	}
}

func (e *Engine) Cancel(id string) bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	if _, ok := e.pending[id]; !ok {
		return false
	}
	e.cancelled[id] = struct{}{}
	return true
}

func (e *Engine) Depth() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.heap.Len()
}

func (e *Engine) Stats() map[string]interface{} {
	e.mu.Lock()
	defer e.mu.Unlock()
	return map[string]interface{}{
		"queue_depth":     e.heap.Len(),
		"total_scheduled": e.scheduled,
		"total_completed": e.completed,
	}
}

func (e *Engine) agingLoop() {
	ticker := time.NewTicker(500 * time.Millisecond)
	for range ticker.C {
		e.mu.Lock()
		now := time.Now()
		for _, req := range e.pending {
			wait := now.Sub(req.EnqueuedAt).Seconds()
			boost := e.agingFactor * wait * 1000
			if boost > 500 {
				boost = 500
			}
			req.Priority = req.PredictedTokens - boost
		}
		e.mu.Unlock()
	}
}

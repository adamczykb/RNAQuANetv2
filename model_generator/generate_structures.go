package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"sync"
)

type Task struct {
	file_path   string
	status      int
	result_path string
}

// Queue represents a simple server state queue
type Queue struct {
	items []Task
	mu    sync.Mutex
}

func NewQueue() *Queue {
	return &Queue{
		items: make([]Task, 0),
	}
}

// Push adds an item to the queue
func (q *Queue) Push(item Task) {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.items = append(q.items, item)
}
func (q *Queue) Len() int {
	q.mu.Lock()
	defer q.mu.Unlock()
	return len(q.items)
}

// WithQueue adds a queue to a context
func WithQueue(ctx context.Context, queue *Queue) context.Context {
	return context.WithValue(ctx, "task_queue", queue)
}

// GetQueue retrieves the queue from a context
func GetQueue(ctx context.Context) (*Queue, bool) {
	queue, ok := ctx.Value("task_queue").(*Queue)
	return queue, ok
}

func setFastaTask(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	queue, ok := GetQueue(ctx)
	if !ok {
		http.Error(w, "No queue found in context", http.StatusInternalServerError)
		return
	}
	fmt.Printf("got / request\n")
	fmt.Printf("%d", queue.Len())
	queue.Push(Task{
		file_path:   "test.fasta",
		status:      0,
		result_path: "result.txt",
	})
	io.WriteString(w, "This is my website!\n")
}

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("/submit/fasta/", setFastaTask)
	ctx, cancelCtx := context.WithCancel(context.Background())
	serverOne := &http.Server{
		Addr:    ":3000",
		Handler: mux,
		BaseContext: func(l net.Listener) context.Context {
			ctx = WithQueue(ctx, NewQueue())
			return ctx
		},
	}
	err := serverOne.ListenAndServe()
	if errors.Is(err, http.ErrServerClosed) {
		fmt.Printf("server closed\n")
	} else if err != nil {
		fmt.Printf("error starting server: %s\n", err)
		os.Exit(1)
	}
	cancelCtx()
	fmt.Printf("server closed\n")
}

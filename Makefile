# Makefile for development automation

.PHONY: help dev start stop restart logs supabase-start supabase-stop supabase-status reset-db backend frontend

help:
	@echo "Available commands:"
	@echo "  make dev          - Start all services via Overmind"
	@echo "  make start        - Alias for 'make dev'"
	@echo "  make stop         - Stop all services"
	@echo "  make restart      - Restart all services"
	@echo "  make logs         - View logs from all processes"
	@echo "  make supabase-start - Start Supabase only"
	@echo "  make supabase-stop  - Stop Supabase only"
	@echo "  make supabase-status - Check Supabase status"
	@echo "  make reset-db     - Reset database and run migrations"
	@echo "  make backend      - Start backend only"
	@echo "  make frontend     - Start frontend only"

dev:
	@echo "Starting all services with Overmind..."
	@if [ -f .overmind.sock ]; then \
		echo "Overmind socket exists, attempting to stop existing instance..."; \
		overmind stop 2>/dev/null || true; \
		sleep 1; \
	fi
	@if tmux has-session -t rag-evaluator 2>/dev/null; then \
		echo "Stopping existing tmux session..."; \
		tmux kill-session -t rag-evaluator 2>/dev/null || true; \
		sleep 1; \
	fi
	@rm -f .overmind.sock
	overmind start -f .overmind

start: dev

stop:
	@echo "Stopping all services..."
	@if [ -f .overmind.sock ]; then \
		overmind stop 2>/dev/null || true; \
	fi
	@echo "Killing any remaining processes..."
	@pkill -f "uvicorn.*rag_eval" 2>/dev/null || true
	@pkill -f "vite" 2>/dev/null || true
	@pkill -f "node.*dev" 2>/dev/null || true
	@# Kill processes on ports 8000 and 5173 if they exist
	@lsof -ti:8000 2>/dev/null | xargs kill -9 2>/dev/null || true
	@lsof -ti:5173 2>/dev/null | xargs kill -9 2>/dev/null || true
	@echo "Services stopped."

restart:
	@echo "Restarting all services..."
	@if [ -f .overmind.sock ]; then \
		overmind restart; \
	else \
		echo "Overmind not running. Use 'make dev' to start services first."; \
		exit 1; \
	fi

logs:
	@echo "Viewing logs..."
	overmind logs

supabase-start:
	@echo "Starting Supabase..."
	cd infra/supabase && supabase start

supabase-stop:
	@echo "Stopping Supabase..."
	cd infra/supabase && supabase stop

supabase-status:
	@echo "Checking Supabase status..."
	@cd infra/supabase && supabase status
	@echo ""
	@echo "=== Full Supabase Keys (JWT Tokens) ==="
	@cd infra/supabase && supabase status --output json 2>/dev/null | python3 -c "import sys, json; data = json.load(sys.stdin); print('  ANON_KEY (Publishable):', data.get('ANON_KEY', 'N/A')); print('  SERVICE_ROLE_KEY (Secret):', data.get('SERVICE_ROLE_KEY', 'N/A'))" 2>/dev/null || echo "  (Run 'cd infra/supabase && supabase status --output json' for full keys)"

reset-db:
	@echo "Resetting database..."
	cd infra/supabase && supabase db reset

backend:
	@echo "Starting backend..."
	cd backend && source venv/bin/activate && PYTHONPATH=backend uvicorn rag_eval.api.main:app --reload --port 8000

frontend:
	@echo "Starting frontend..."
	cd frontend && npm run dev


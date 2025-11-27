# Makefile for development automation

.PHONY: help dev stop restart logs supabase-start supabase-stop reset-db backend frontend

help:
	@echo "Available commands:"
	@echo "  make dev          - Start all services via Overmind"
	@echo "  make stop         - Stop all services"
	@echo "  make restart      - Restart all services"
	@echo "  make logs         - View logs from all processes"
	@echo "  make supabase-start - Start Supabase only"
	@echo "  make supabase-stop  - Stop Supabase only"
	@echo "  make reset-db     - Reset database and run migrations"
	@echo "  make backend      - Start backend only"
	@echo "  make frontend     - Start frontend only"

dev:
	@echo "Starting all services with Overmind..."
	cd .. && overmind start -f .overmind

stop:
	@echo "Stopping all services..."
	cd .. && overmind stop

restart:
	@echo "Restarting all services..."
	cd .. && overmind restart

logs:
	@echo "Viewing logs..."
	cd .. && overmind logs

supabase-start:
	@echo "Starting Supabase..."
	cd supabase && supabase start

supabase-stop:
	@echo "Stopping Supabase..."
	cd supabase && supabase stop

reset-db:
	@echo "Resetting database..."
	cd supabase && supabase db reset

backend:
	@echo "Starting backend..."
	cd ../backend && uvicorn rag_eval.api.main:app --reload --port 8000

frontend:
	@echo "Starting frontend..."
	cd ../frontend && npm run dev


.PHONY: dev dev-frontend dev-backend

dev:
	@echo "Starting both frontend and backend for local development..."
	@make -j 2 dev-frontend dev-backend

dev-frontend:
	cd frontend && npm run dev

dev-backend:
	cd backend && python -m flask --app "app:create_app()" run --debug --port=5000

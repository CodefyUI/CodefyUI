RUNNER := ./cdui

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "CodefyUI"
	@echo ""
	@echo "  make install   安裝所有依賴（backend + frontend）"
	@echo "  make dev       啟動開發伺服器（backend :8000 + frontend :5173）"
	@echo "  make stop      停止所有服務"
	@echo "  make test      執行 backend + frontend 測試（無 pnpm 時略過前端）"
	@echo "  make clean     移除虛擬環境與 node_modules"
	@echo ""
	@echo "或直接："
	@echo "  ./cdui <command>       # macOS / Linux"
	@echo "  .\\cdui.cmd <command>   # Windows"
	@echo ""

.PHONY: install
install:
	$(RUNNER) install

.PHONY: dev
dev:
	$(RUNNER) dev

.PHONY: stop
stop:
	$(RUNNER) stop

.PHONY: test
test:
	$(RUNNER) test

.PHONY: clean
clean:
	$(RUNNER) clean

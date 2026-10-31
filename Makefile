export PYTHONPATH := .
.PHONY: test blockchain k8s-setup k8s llm-proxy clean

test:
	python tests/run_tests.py

blockchain:
	python -m fairattack.blockchain.cli --nodes 4 --rounds 3 --txs 5

k8s-setup:
	bash k8s/setup.sh

k8s:
	bash scripts/run_k8s_suite.sh

llm-proxy:
	python llm_gpu/vllm_vtc_proxy.py --port 9000 --max-inflight 64

clean:
	rm -f results/*.csv results/*.json figs/*.png
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

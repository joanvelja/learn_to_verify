.PHONY: hooks
hooks:
	pre-commit install --overwrite --install-hooks --hook-type pre-commit --hook-type post-checkout --hook-type pre-push
	git checkout master
	echo "Installing pre-commit hook..."
	chmod +x .git/hooks/pre-commit
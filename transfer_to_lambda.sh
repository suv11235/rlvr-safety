#!/bin/bash
# Transfer Token-Buncher to Lambda GPU

LAMBDA_IP=$1
LAMBDA_USER=${2:-ubuntu}
SSH_KEY=${3:-~/.ssh/lambda_key}
LAMBDA_DIR=${4:-mars/suvajit}  # Default to mars/suvajit

if [ -z "$LAMBDA_IP" ]; then
    echo "Usage: $0 <lambda-ip> [user] [ssh-key] [target-dir]"
    echo "Example: $0 123.45.67.89 ubuntu ~/.ssh/lambda_key mars/suvajit"
    exit 1
fi

echo "Creating archive..."
cd "$(dirname "$0")"
tar -czf /tmp/token-buncher-lambda.tar.gz \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='outputs' \
    --exclude='*.log' \
    --exclude='dataset/beavertails-qwen' \
    --exclude='dataset/wmdp*' \
    --exclude='dataset/gsm8k*' \
    .

echo "Transferring to Lambda..."
if [ -f "$SSH_KEY" ]; then
    scp -i "$SSH_KEY" /tmp/token-buncher-lambda.tar.gz ${LAMBDA_USER}@${LAMBDA_IP}:~/
else
    scp /tmp/token-buncher-lambda.tar.gz ${LAMBDA_USER}@${LAMBDA_IP}:~/
fi

echo "Extracting to $LAMBDA_DIR on Lambda..."
if [ -f "$SSH_KEY" ]; then
    ssh -i "$SSH_KEY" ${LAMBDA_USER}@${LAMBDA_IP} "mkdir -p $LAMBDA_DIR && cd $LAMBDA_DIR && tar -xzf ~/token-buncher-lambda.tar.gz && rm ~/token-buncher-lambda.tar.gz && echo 'Transfer complete!'"
else
    ssh ${LAMBDA_USER}@${LAMBDA_IP} "mkdir -p $LAMBDA_DIR && cd $LAMBDA_DIR && tar -xzf ~/token-buncher-lambda.tar.gz && rm ~/token-buncher-lambda.tar.gz && echo 'Transfer complete!'"
fi

echo ""
echo "Done! Connect to Lambda and run:"
echo "  cd ~/$LAMBDA_DIR/Token-Buncher && bash setup_lambda.sh"
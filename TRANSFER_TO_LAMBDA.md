# Transferring Token-Buncher to Lambda GPU

This guide covers the best methods to transfer the Token-Buncher repository to your Lambda GPU instance.

## Method 1: Direct Download on Lambda (Recommended - Simplest)

**Best for**: Quick setup, no local file transfer needed

```bash
# On your Lambda GPU instance
cd ~
curl -L "https://anonymous.4open.science/api/repo/Token-Buncher/zip" -o Token-Buncher.zip
unzip Token-Buncher.zip
cd Token-Buncher

# Then run setup
bash setup_lambda.sh
```

**Pros**:
- No file transfer needed
- Fastest initial setup
- Works from anywhere

**Cons**:
- Won't include your custom preprocessing scripts (need to add separately)
- Need to recreate dataset directory structure

## Method 2: SCP Transfer (Best for Custom Files)

**Best for**: Transferring your prepared directory with all custom scripts

### Step 1: Create a compressed archive locally

```bash
# On your local machine (from tamperdefense directory)
cd imports/Token-Buncher
tar -czf token-buncher-lambda.tar.gz \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='outputs' \
    --exclude='*.log' \
    .
```

### Step 2: Transfer via SCP

```bash
# On your local machine
scp token-buncher-lambda.tar.gz ubuntu@<lambda-ip>:~/

# Or if using Lambda's SSH key format:
scp -i ~/.ssh/lambda_key token-buncher-lambda.tar.gz ubuntu@<lambda-ip>:~/
```

### Step 3: Extract on Lambda

```bash
# On Lambda GPU instance
cd ~
tar -xzf token-buncher-lambda.tar.gz
cd Token-Buncher
bash setup_lambda.sh
```

**Pros**:
- Includes all your custom preprocessing scripts
- Includes setup scripts and documentation
- One-time transfer

**Cons**:
- Requires SSH access configured
- File size may be large (but compressed)

## Method 3: rsync (Best for Updates)

**Best for**: Syncing changes, incremental updates

```bash
# On your local machine
rsync -avz --exclude '__pycache__' \
           --exclude '*.pyc' \
           --exclude 'outputs' \
           --exclude '*.log' \
           --exclude '.git' \
           imports/Token-Buncher/ \
           ubuntu@<lambda-ip>:~/Token-Buncher/
```

**Pros**:
- Efficient for updates
- Only transfers changed files
- Can resume interrupted transfers

**Cons**:
- Requires SSH access
- More complex command

## Method 4: Git Repository (Best for Version Control)

**Best for**: If you want to maintain version control

### Step 1: Create a git repo (locally or on GitHub)

```bash
# On your local machine
cd imports/Token-Buncher
git init
git add .
git commit -m "Initial Token-Buncher setup for Lambda"
git remote add origin <your-git-repo-url>
git push -u origin main
```

### Step 2: Clone on Lambda

```bash
# On Lambda GPU instance
cd ~
git clone <your-git-repo-url> Token-Buncher
cd Token-Buncher
bash setup_lambda.sh
```

**Pros**:
- Version control
- Easy updates via `git pull`
- Can share with team

**Cons**:
- Requires git repository setup
- Need to handle large files (git LFS if needed)

## Method 5: Lambda Web Interface (If Available)

Some Lambda instances provide web-based file upload:
- Check Lambda dashboard for file upload feature
- Upload the tar.gz archive
- Extract on the instance

## Recommended Approach

**For first-time setup**: Use **Method 2 (SCP)** to transfer your prepared directory with all custom scripts and documentation.

**For updates**: Use **Method 3 (rsync)** or **Method 4 (Git)**.

## Quick Transfer Script

Save this as `transfer_to_lambda.sh` on your local machine:

```bash
#!/bin/bash
# Transfer Token-Buncher to Lambda GPU

LAMBDA_IP=$1
LAMBDA_USER=${2:-ubuntu}
SSH_KEY=${3:-~/.ssh/lambda_key}

if [ -z "$LAMBDA_IP" ]; then
    echo "Usage: $0 <lambda-ip> [user] [ssh-key]"
    echo "Example: $0 123.45.67.89 ubuntu ~/.ssh/lambda_key"
    exit 1
fi

echo "Creating archive..."
cd imports/Token-Buncher
tar -czf /tmp/token-buncher-lambda.tar.gz \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='outputs' \
    --exclude='*.log' \
    .

echo "Transferring to Lambda..."
if [ -f "$SSH_KEY" ]; then
    scp -i "$SSH_KEY" /tmp/token-buncher-lambda.tar.gz ${LAMBDA_USER}@${LAMBDA_IP}:~/
else
    scp /tmp/token-buncher-lambda.tar.gz ${LAMBDA_USER}@${LAMBDA_IP}:~/
fi

echo "Extracting on Lambda..."
if [ -f "$SSH_KEY" ]; then
    ssh -i "$SSH_KEY" ${LAMBDA_USER}@${LAMBDA_IP} "cd ~ && tar -xzf token-buncher-lambda.tar.gz && rm token-buncher-lambda.tar.gz"
else
    ssh ${LAMBDA_USER}@${LAMBDA_IP} "cd ~ && tar -xzf token-buncher-lambda.tar.gz && rm token-buncher-lambda.tar.gz"
fi

echo "Done! Connect to Lambda and run: cd ~/Token-Buncher && bash setup_lambda.sh"
```

Usage:
```bash
chmod +x transfer_to_lambda.sh
./transfer_to_lambda.sh <lambda-ip>
```

## Verification After Transfer

Once transferred, verify on Lambda:

```bash
# On Lambda GPU instance
cd ~/Token-Buncher
ls -la
# Should see: setup_lambda.sh, dataset/, configs/, verl/, etc.

# Verify preprocessing scripts
ls dataset/data_preprocess/
# Should see: beavertails.py, wmdp.py, wmdp_merge.py, gsm8k.py

# Check setup script
bash setup_lambda.sh
```

## File Size Estimates

- **Compressed (tar.gz)**: ~5-10 MB (code only)
- **Uncompressed**: ~50-100 MB (code + structure)
- **With datasets**: Will be larger (datasets downloaded separately)

## Notes

- **SSH Access**: Ensure your SSH key is added to Lambda instance
- **Network Speed**: SCP/rsync speed depends on your connection
- **Firewall**: Ensure Lambda instance allows SSH (port 22)
- **Permissions**: May need to `chmod +x` scripts after transfer

## Troubleshooting

### SSH Connection Issues
```bash
# Test SSH connection
ssh ubuntu@<lambda-ip>

# If using key
ssh -i ~/.ssh/lambda_key ubuntu@<lambda-ip>
```

### Permission Denied
```bash
# On Lambda, fix permissions
chmod +x setup_lambda.sh
chmod +x dataset/data_preprocess/*.py
```

### Large File Transfer
```bash
# Use compression and resume capability
rsync -avz --progress imports/Token-Buncher/ ubuntu@<lambda-ip>:~/Token-Buncher/
```


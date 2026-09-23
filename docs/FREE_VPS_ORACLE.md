# Free VPS deployment (Oracle Cloud Always Free)

This guide assumes:

- Ubuntu 22.04 or 24.04
- Docker + Docker Compose installed
- A free DuckDNS domain pointing to the VPS public IP
- Ports 80 and 443 open in both the cloud firewall and the OS firewall

## 1. Install Docker

```bash
sudo apt update
sudo apt install -y docker.io docker-compose-plugin git
sudo usermod -aG docker "$USER"
newgrp docker
```

## 2. Clone the repository

```bash
git clone https://github.com/runwindy/aml-text-memory.git
cd aml-text-memory
```

## 3. Create the production env file

```bash
cp deploy/.env.prod.example .env
nano .env
```

Replace:

- `AML_MEMORY_SYSTEM_KEY`
- `AML_EMBEDDING_API_KEY`

## 4. Configure Caddy

```bash
cp deploy/Caddyfile.example deploy/Caddyfile
nano deploy/Caddyfile
```

Replace `your-domain.example.com` with your DuckDNS domain.

## 5. Open the OS firewall

```bash
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

Also open ports 80 and 443 in the Oracle Cloud Security List / Network Security Group.

## 6. Start the service

```bash
docker compose -f deploy/docker-compose.prod.yml up -d --build
```

## 7. Check logs

```bash
docker compose -f deploy/docker-compose.prod.yml logs -f
```

## 8. Test

```bash
curl https://your-domain.example.com/health
```

Expected:

```json
{"status":"ok"}
```

Then run the local smoke test against the public domain:

```powershell
python scripts\local_smoke.py --base-url https://your-domain.example.com --key YOUR_MEMORY_SYSTEM_KEY
```

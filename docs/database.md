# Database Guide

This project uses **PostgreSQL 16** for both local development and production.

| Environment | Engine | Host |
|-------------|--------|------|
| Local dev | Docker (postgres:16-alpine) | `localhost:5432` |
| Production | Supabase (PostgreSQL 16) | Supabase connection string |

---

## Starting and Stopping

```bash
# Start PostgreSQL + pgAdmin in the background
docker compose up -d

# Check container status (both should show "healthy")
docker ps

# Follow postgres logs in real time
docker compose logs -f postgres

# Stop containers — data is preserved in named volumes
docker compose down

# Stop AND wipe all data (full reset — WARNING: irreversible)
docker compose down -v
```

---

## Connecting via pgAdmin (GUI)

1. Open **http://localhost:5050**
2. Login with `PGADMIN_DEFAULT_EMAIL` and `PGADMIN_DEFAULT_PASSWORD` from your `.env`
3. Add the database server:
   - Right-click **Servers** → **Register → Server**
   - **General tab → Name:** `FMD Local`
   - **Connection tab:**
     | Field | Value |
     |-------|-------|
     | Host | `fmd-postgres` (Docker service name — not `localhost`) |
     | Port | `5432` |
     | Maintenance DB | value of `POSTGRES_DB` in `.env` |
     | Username | value of `POSTGRES_USER` in `.env` |
     | Password | value of `POSTGRES_PASSWORD` in `.env` |
4. Click **Save**

> **Why `fmd-postgres` and not `localhost`?** pgAdmin runs inside Docker on the same network as the PostgreSQL container. Inside Docker, services resolve by their container name, not `localhost`.

---

## Connecting via CLI (psql)

```bash
# Open an interactive psql session inside the container
docker exec -it fmd-postgres psql -U fmd_user -d face_morphing_db

# Run a single SQL command and exit
docker exec fmd-postgres psql -U fmd_user -d face_morphing_db -c "SELECT version();"

# List all tables
docker exec fmd-postgres psql -U fmd_user -d face_morphing_db -c "\dt"
```

---

## Backup and Restore

### Create a backup (pg_dump)

```bash
docker exec fmd-postgres pg_dump -U fmd_user face_morphing_db > backup.sql
```

This writes a plain-SQL dump to `backup.sql` in your current directory.

### Restore from a backup

```bash
cat backup.sql | docker exec -i fmd-postgres psql -U fmd_user -d face_morphing_db
```

### Compressed backup (recommended for larger DBs)

```bash
# Dump in custom format (smaller, faster restore)
docker exec fmd-postgres pg_dump -U fmd_user -Fc face_morphing_db > backup.dump

# Restore from custom format
docker exec -i fmd-postgres pg_restore -U fmd_user -d face_morphing_db < backup.dump
```

---

## Resetting All Data

```bash
# Stops containers and deletes named volumes (fmd-postgres-data, fmd-pgadmin-data)
docker compose down -v

# Start fresh
docker compose up -d
```

---

## Troubleshooting

### Port 5432 already in use

Another PostgreSQL instance is running locally on port 5432.

```bash
# Check what's using the port
netstat -ano | findstr :5432         # Windows
lsof -i :5432                        # Mac/Linux
```

**Fix:** Either stop the local PostgreSQL service, or change `POSTGRES_PORT` in your `.env`:
```env
POSTGRES_PORT=5433
```
Then update `DATABASE_URL` to match:
```env
DATABASE_URL=postgresql://fmd_user:yourpassword@localhost:5433/face_morphing_db
```

---

### Port 5050 already in use (pgAdmin)

Change `PGADMIN_PORT` in your `.env`:
```env
PGADMIN_PORT=5051
```

---

### Postgres container exits immediately

```bash
# Check logs for the error
docker compose logs postgres
```

Common causes:
- Missing or empty `.env` — `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` are required
- Corrupted volume data — run `docker compose down -v` to wipe and restart

---

### pgAdmin keeps restarting — email validation error

pgAdmin's newer versions reject **reserved TLDs** in the login email. The following TLDs are blocked by RFC standards and will cause a restart loop:

- `.local` (reserved for mDNS / Bonjour — RFC 6762)
- `.test`, `.invalid`, `.example`, `.localhost` (reserved — RFC 2606)

**Symptom:** `docker ps` shows `fmd-pgadmin` with status `Restarting (1) X seconds ago`.

**Fix:** Use a real-looking domain in `PGADMIN_DEFAULT_EMAIL`:
```env
# Bad — causes restart loop
PGADMIN_DEFAULT_EMAIL=admin@fmd.local

# Good
PGADMIN_DEFAULT_EMAIL=admin@example.com
```

After changing `.env`, restart the containers:
```bash
docker compose down && docker compose up -d
```

---

### pgAdmin container shows "unhealthy" or won't start

pgAdmin depends on postgres being `healthy`. If postgres isn't healthy yet, pgAdmin won't start. Wait 30 seconds and re-check:

```bash
docker ps
```

If postgres is still unhealthy:
```bash
docker compose logs postgres
```

---

### Can't connect from pgAdmin — "Connection refused"

Make sure you're using **`fmd-postgres`** as the host (not `localhost`). pgAdmin runs inside Docker and resolves the database by container name on the `fmd-network` bridge.

---

## Environment Variables Reference

| Variable | Example | Used By |
|----------|---------|---------|
| `POSTGRES_USER` | `fmd_user` | docker-compose, Prisma, SQLAlchemy |
| `POSTGRES_PASSWORD` | `changeme` | docker-compose, DATABASE_URL |
| `POSTGRES_DB` | `face_morphing_db` | docker-compose, DATABASE_URL |
| `POSTGRES_PORT` | `5432` | docker-compose port mapping |
| `PGADMIN_DEFAULT_EMAIL` | `admin@example.com` | pgAdmin login |
| `PGADMIN_DEFAULT_PASSWORD` | `changeme` | pgAdmin login |
| `PGADMIN_PORT` | `5050` | pgAdmin host port |
| `DATABASE_URL` | `postgresql://...` | Prisma (`frontend`), SQLAlchemy (`backend`) |

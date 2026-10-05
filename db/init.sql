CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS btree_gist;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TYPE reservation_status AS ENUM ('CONFIRMED','PICKED_UP','RETURNED','CANCELLED');
CREATE TYPE payment_status     AS ENUM ('PENDING','AUTHORIZED','PAID','REFUNDED');

CREATE TABLE customers (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email         TEXT UNIQUE NOT NULL,
  full_name     TEXT NOT NULL,
  phone         TEXT,
  password_hash TEXT NOT NULL,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE locations (
  id        SERIAL PRIMARY KEY,
  code      TEXT UNIQUE NOT NULL,
  name      TEXT NOT NULL,
  city      TEXT NOT NULL,
  country   CHAR(2) NOT NULL,
  timezone  TEXT NOT NULL,
  aliases   TEXT[] NOT NULL DEFAULT '{}'
);

CREATE TABLE vehicle_models (
  id           SERIAL PRIMARY KEY,
  make         TEXT NOT NULL,
  model        TEXT NOT NULL,
  category     TEXT NOT NULL,
  seats        SMALLINT NOT NULL,
  large_bags   SMALLINT NOT NULL,
  small_bags   SMALLINT NOT NULL,
  transmission TEXT NOT NULL,
  fuel_type    TEXT NOT NULL,
  features     JSONB NOT NULL DEFAULT '{}'
);

CREATE TABLE vehicles (
  id               SERIAL PRIMARY KEY,
  model_id         INT NOT NULL REFERENCES vehicle_models(id),
  home_location_id INT NOT NULL REFERENCES locations(id),
  plate            TEXT UNIQUE NOT NULL,
  status           TEXT NOT NULL DEFAULT 'ACTIVE'
);

CREATE TABLE rates (
  id          SERIAL PRIMARY KEY,
  location_id INT NOT NULL REFERENCES locations(id),
  category    TEXT NOT NULL,
  daily_rate  NUMERIC(10,2) NOT NULL,
  currency    CHAR(3) NOT NULL,
  valid_from  DATE NOT NULL,
  valid_to    DATE NOT NULL
);

CREATE TABLE insurance_plans (
  code        TEXT PRIMARY KEY,
  name        TEXT NOT NULL,
  daily_price NUMERIC(10,2) NOT NULL,
  deductible  NUMERIC(10,2) NOT NULL,
  currency    CHAR(3) NOT NULL,
  kb_doc_id   TEXT NOT NULL
);

CREATE TABLE reservations (
  id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  confirmation_code  TEXT UNIQUE NOT NULL,
  customer_id        UUID NOT NULL REFERENCES customers(id),
  vehicle_id         INT  NOT NULL REFERENCES vehicles(id),
  pickup_location_id INT  NOT NULL REFERENCES locations(id),
  return_location_id INT  NOT NULL REFERENCES locations(id),
  pickup_at          TIMESTAMPTZ NOT NULL,
  return_at          TIMESTAMPTZ NOT NULL,
  rental_period      TSTZRANGE GENERATED ALWAYS AS (tstzrange(pickup_at, return_at)) STORED,
  status             reservation_status NOT NULL,
  payment_status     payment_status NOT NULL,
  total_amount       NUMERIC(10,2) NOT NULL,
  currency           CHAR(3) NOT NULL,
  version            INT NOT NULL DEFAULT 1,
  created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
  CHECK (return_at > pickup_at),
  EXCLUDE USING gist (vehicle_id WITH =, rental_period WITH &&)
    WHERE (status IN ('CONFIRMED','PICKED_UP'))
);
CREATE INDEX idx_reservations_customer ON reservations (customer_id, pickup_at);

CREATE TABLE reservation_insurance (
  reservation_id UUID NOT NULL REFERENCES reservations(id),
  plan_code      TEXT NOT NULL REFERENCES insurance_plans(code),
  price          NUMERIC(10,2) NOT NULL,
  PRIMARY KEY (reservation_id, plan_code)
);

CREATE TABLE reservation_changes (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  reservation_id  UUID NOT NULL REFERENCES reservations(id),
  customer_id     UUID NOT NULL REFERENCES customers(id),
  change_type     TEXT NOT NULL,
  old_values      JSONB NOT NULL,
  new_values      JSONB NOT NULL,
  price_delta     NUMERIC(10,2) NOT NULL,
  idempotency_key TEXT UNIQUE NOT NULL,
  status          TEXT NOT NULL,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE kb_documents (
  doc_id     TEXT PRIMARY KEY,
  title      TEXT NOT NULL,
  category   TEXT NOT NULL,      -- INSURANCE | ACCIDENT | CANCELLATION | ROADSIDE | TERMS
  version    TEXT NOT NULL,
  source_uri TEXT
);

CREATE TABLE kb_chunks (
  id         BIGSERIAL PRIMARY KEY,
  doc_id     TEXT NOT NULL REFERENCES kb_documents(doc_id),
  section    TEXT NOT NULL,
  content    TEXT NOT NULL,
  plan_codes TEXT[] NOT NULL DEFAULT '{}',
  embedding  vector(384) NOT NULL
);
CREATE INDEX idx_kb_chunks_embedding ON kb_chunks USING hnsw (embedding vector_cosine_ops);

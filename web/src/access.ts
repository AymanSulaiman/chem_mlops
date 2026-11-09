// Demo access control: who is calling (Sign in with Google), how many people
// are in at once, and how much each may chat. All state is in memory, which is
// enough for the single-instance demo; it resets when the server restarts.

import { createHmac, randomBytes, timingSafeEqual } from "node:crypto";

// ── Google ID token ─────────────────────────────────────────────────────────

type Jwk = JsonWebKey & { kid?: string };
type Fetch = (input: URL | RequestInfo, init?: RequestInit) => Promise<Response>;

export const GOOGLE_CERTS_URL = "https://www.googleapis.com/oauth2/v3/certs";
const GOOGLE_ISSUERS = ["https://accounts.google.com", "accounts.google.com"];
const KEY_REFRESH_MS = 60_000; // an unknown kid refetches the certs at most this often

const decodePart = (part: string) => JSON.parse(Buffer.from(part, "base64url").toString("utf8"));

// Verifies the ID token the Google sign-in button hands the page, and returns
// the visitor's email, or null if the token is forged, expired, issued for
// another app, or the email isn't verified.
export function createGoogleVerifier(clientId: string, fetchImpl: Fetch = fetch, now = () => Date.now()) {
  let keys = new Map<string, CryptoKey>();
  let fetchedAt = -Infinity;

  async function refreshKeys() {
    fetchedAt = now();
    const res = await fetchImpl(GOOGLE_CERTS_URL);
    if (!res.ok) throw new Error(`Could not fetch Google certs: ${res.status}`);
    const { keys: jwks = [] } = (await res.json()) as { keys?: Jwk[] };
    const next = new Map<string, CryptoKey>();
    for (const jwk of jwks) {
      if (!jwk.kid || jwk.kty !== "RSA") continue;
      next.set(
        jwk.kid,
        await crypto.subtle.importKey("jwk", jwk, { name: "RSASSA-PKCS1-v1_5", hash: "SHA-256" }, false, ["verify"]),
      );
    }
    keys = next;
  }

  async function keyFor(kid: string): Promise<CryptoKey | undefined> {
    if (!keys.has(kid) && now() - fetchedAt > KEY_REFRESH_MS) await refreshKeys();
    return keys.get(kid);
  }

  return async function verify(token: unknown): Promise<string | null> {
    if (typeof token !== "string") return null;
    const parts = token.split(".");
    if (parts.length !== 3) return null;
    const [head, body, sig] = parts as [string, string, string];

    let header: { alg?: string; kid?: string };
    let claims: {
      aud?: string | string[];
      iss?: string;
      exp?: number;
      nbf?: number;
      email?: string;
      email_verified?: boolean | string;
    };
    try {
      header = decodePart(head);
      claims = decodePart(body);
    } catch {
      return null;
    }
    if (header.alg !== "RS256" || !header.kid) return null;

    const key = await keyFor(header.kid);
    if (!key) return null;
    const valid = await crypto.subtle.verify(
      "RSASSA-PKCS1-v1_5",
      key,
      Buffer.from(sig, "base64url"),
      new TextEncoder().encode(`${head}.${body}`),
    );
    if (!valid) return null;

    const seconds = now() / 1000;
    const audiences = Array.isArray(claims.aud) ? claims.aud : [claims.aud];
    if (!audiences.includes(clientId)) return null;
    if (!claims.iss || !GOOGLE_ISSUERS.includes(claims.iss)) return null;
    if (typeof claims.exp !== "number" || claims.exp <= seconds) return null;
    if (typeof claims.nbf === "number" && claims.nbf > seconds) return null;
    if (claims.email_verified !== true && claims.email_verified !== "true") return null;
    if (typeof claims.email !== "string" || !claims.email) return null;
    return claims.email.toLowerCase();
  };
}

// ── Session cookie ──────────────────────────────────────────────────────────
// After sign-in the server sets `session=<email>|<expiry>|<hmac>`. The key is
// random per process: a restart (e.g. a redeploy) signs everyone out, and they
// sign back in with one click.

export const SESSION_COOKIE = "session";
const SESSION_MS = 24 * 60 * 60_000;

export function createSessions(now = () => Date.now(), key: Buffer = randomBytes(32)) {
  const sign = (payload: string) => createHmac("sha256", key).update(payload).digest("base64url");

  return {
    // The Set-Cookie header value for a new session.
    issue(email: string): string {
      const payload = `${encodeURIComponent(email)}|${now() + SESSION_MS}`;
      return `${SESSION_COOKIE}=${payload}|${sign(payload)}; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=${SESSION_MS / 1000}`;
    },
    clear(): string {
      return `${SESSION_COOKIE}=; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=0`;
    },
    // The signed-in email from a request's Cookie header, or null.
    read(cookieHeader: string | null): string | null {
      const raw = cookieHeader
        ?.split(";")
        .map((part) => part.trim())
        .find((part) => part.startsWith(`${SESSION_COOKIE}=`))
        ?.slice(SESSION_COOKIE.length + 1);
      if (!raw) return null;
      const cut = raw.lastIndexOf("|");
      if (cut === -1) return null;
      const payload = raw.slice(0, cut);
      const given = Buffer.from(raw.slice(cut + 1));
      const expected = Buffer.from(sign(payload));
      if (given.length !== expected.length || !timingSafeEqual(given, expected)) return null;
      const [email, expiry] = payload.split("|");
      if (!email || !(Number(expiry) > now())) return null;
      return decodeURIComponent(email);
    },
  };
}

// ── Queue and rate limits ───────────────────────────────────────────────────
// People join a queue in arrival order; the first `maxUsers` are in the chat,
// everyone else sees their place. Everyone pings to keep their place; someone
// who stops pinging (closed the tab) drops out after `idleMs`. While anyone is
// waiting, a person in the chat who hasn't sent a message for `turnMs` gives up
// their seat, so a forgotten open tab can't hold the demo forever.

export type GateOptions = {
  maxUsers?: number;
  idleMs?: number;
  turnMs?: number;
  chatsPerHour?: number;
  chatsPerDay?: number;
  now?: () => number;
};

// 0 = in the chat; N = Nth in line.
export type Place = { seated: boolean; position: number };

const HOUR = 60 * 60_000;
const DAY = 24 * HOUR;

export function createGate(options: GateOptions = {}) {
  const maxUsers = options.maxUsers ?? 1;
  const idleMs = options.idleMs ?? 2 * 60_000;
  const turnMs = options.turnMs ?? 5 * 60_000;
  const chatsPerHour = options.chatsPerHour ?? 20;
  const chatsPerDay = options.chatsPerDay ?? 100;
  const now = options.now ?? (() => Date.now());

  let line: string[] = []; // arrival order; the first maxUsers are seated
  const lastSeen = new Map<string, number>(); // email -> last ping or request
  const lastActive = new Map<string, number>(); // seated email -> seated at / last chat
  const chats = new Map<string, number[]>(); // email -> chat start times, last 24h

  function remove(email: string) {
    line = line.filter((e) => e !== email);
    lastSeen.delete(email);
    lastActive.delete(email);
  }

  // Drop people who left, end idle turns while others wait, seat whoever is next.
  function settle(t: number) {
    for (const email of [...line]) if (t - (lastSeen.get(email) ?? 0) > idleMs) remove(email);
    if (line.length > maxUsers) {
      for (const email of line.slice(0, maxUsers)) {
        if (t - (lastActive.get(email) ?? t) > turnMs) {
          // Back of the line, not out: they can keep waiting for another turn.
          line = [...line.filter((e) => e !== email), email];
          lastActive.delete(email);
        }
      }
    }
    for (const email of line.slice(0, maxUsers)) if (!lastActive.has(email)) lastActive.set(email, t);
  }

  function placeOf(email: string): Place {
    const index = line.indexOf(email);
    return index < maxUsers ? { seated: true, position: 0 } : { seated: false, position: index - maxUsers + 1 };
  }

  return {
    maxUsers,
    // Join the line, or keep your place in it.
    claim(email: string): Place {
      const t = now();
      if (!line.includes(email)) line.push(email);
      lastSeen.set(email, t);
      settle(t);
      return placeOf(email);
    },
    leave(email: string) {
      remove(email);
      settle(now());
    },
    waiting(): number {
      settle(now());
      return Math.max(0, line.length - maxUsers);
    },
    // Records a chat if the person is under both limits; a chat also restarts
    // their turn timer.
    allowChat(email: string): boolean {
      const t = now();
      const recent = (chats.get(email) ?? []).filter((start) => t - start < DAY);
      const lastHour = recent.filter((start) => t - start < HOUR).length;
      if (lastHour >= chatsPerHour || recent.length >= chatsPerDay) {
        chats.set(email, recent);
        return false;
      }
      recent.push(t);
      chats.set(email, recent);
      if (lastActive.has(email)) lastActive.set(email, t);
      return true;
    },
  };
}

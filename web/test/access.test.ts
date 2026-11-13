import { expect, test } from "bun:test";

import { createGate, createGoogleVerifier, createSessions, GOOGLE_CERTS_URL } from "../src/access";
import { createChatRequestHandler } from "../src/app";
import { MAX_MESSAGES, MAX_USER_CHARS } from "../src/frontend-helpers";

const CLIENT_ID = "test-client.apps.googleusercontent.com";
const NOW = 1_800_000_000_000;

const b64url = (data: string) => Buffer.from(data).toString("base64url");

async function keyPair(kid: string) {
  const pair = await crypto.subtle.generateKey(
    { name: "RSASSA-PKCS1-v1_5", modulusLength: 2048, publicExponent: new Uint8Array([1, 0, 1]), hash: "SHA-256" },
    true,
    ["sign", "verify"],
  );
  const jwk = { ...(await crypto.subtle.exportKey("jwk", pair.publicKey)), kid };
  return { privateKey: pair.privateKey, jwk };
}

const signer = await keyPair("key-1");
const stranger = await keyPair("key-1"); // same kid, different key: a forgery

// A Google ID token as the sign-in button would hand it over.
async function token(claims: Record<string, unknown>, key = signer.privateKey, header: Record<string, unknown> = {}) {
  const head = b64url(JSON.stringify({ alg: "RS256", kid: "key-1", ...header }));
  const body = b64url(
    JSON.stringify({
      aud: CLIENT_ID,
      iss: "https://accounts.google.com",
      exp: NOW / 1000 + 3600,
      email: "Visitor@Example.com",
      email_verified: true,
      ...claims,
    }),
  );
  const sig = await crypto.subtle.sign("RSASSA-PKCS1-v1_5", key, new TextEncoder().encode(`${head}.${body}`));
  return `${head}.${body}.${Buffer.from(sig).toString("base64url")}`;
}

function certsFetch(onOther?: (url: string) => Response | Promise<Response>) {
  return async (input: URL | RequestInfo) => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    if (url === GOOGLE_CERTS_URL) return Response.json({ keys: [signer.jwk] });
    if (onOther) return onOther(url);
    throw new Error(`Unexpected request: ${url}`);
  };
}

const verify = createGoogleVerifier(CLIENT_ID, certsFetch(), () => NOW);

test("a valid Google ID token gives the visitor's email, lowercased", async () => {
  expect(await verify(await token({}))).toBe("visitor@example.com");
  expect(await verify(await token({ iss: "accounts.google.com" }))).toBe("visitor@example.com");
});

test("forged, expired, misaddressed or unverified tokens are rejected", async () => {
  expect(await verify(null)).toBeNull();
  expect(await verify("not.a.jwt")).toBeNull();
  expect(await verify(await token({}, stranger.privateKey))).toBeNull();
  expect(await verify(await token({ exp: NOW / 1000 - 1 }))).toBeNull();
  expect(await verify(await token({ nbf: NOW / 1000 + 60 }))).toBeNull();
  expect(await verify(await token({ aud: "another-app" }))).toBeNull();
  expect(await verify(await token({ iss: "https://evil.example.com" }))).toBeNull();
  expect(await verify(await token({ email_verified: false }))).toBeNull();
  expect(await verify(await token({ email: undefined }))).toBeNull();
  expect(await verify(await token({}, signer.privateKey, { alg: "none" }))).toBeNull();
  expect(await verify(await token({}, signer.privateKey, { kid: "unknown" }))).toBeNull();
});

test("session cookies round-trip, and tampered or expired ones are rejected", () => {
  let t = NOW;
  const sessions = createSessions(() => t);
  const cookie = sessions.issue("a@x.com").split(";")[0]!;
  expect(sessions.read(`other=1; ${cookie}`)).toBe("a@x.com");

  expect(sessions.read(cookie.replace("a%40x.com", "b%40x.com"))).toBeNull();
  expect(sessions.read(`${cookie}x`)).toBeNull();
  expect(createSessions(() => t).read(cookie)).toBeNull(); // another server's key
  expect(sessions.read(null)).toBeNull();

  t += 25 * 60 * 60_000;
  expect(sessions.read(cookie)).toBeNull();
});

test("one person is in the chat and the rest get their number in line", () => {
  let t = 0;
  const gate = createGate({ idleMs: 120_000, now: () => t });
  expect(gate.claim("a@x.com")).toEqual({ seated: true, position: 0 });
  expect(gate.claim("b@x.com")).toEqual({ seated: false, position: 1 });
  expect(gate.claim("c@x.com")).toEqual({ seated: false, position: 2 });
  expect(gate.claim("a@x.com")).toEqual({ seated: true, position: 0 }); // pinging keeps the seat
  expect(gate.waiting()).toBe(2);

  gate.leave("a@x.com"); // closed the tab
  expect(gate.claim("b@x.com")).toEqual({ seated: true, position: 0 });
  expect(gate.claim("c@x.com")).toEqual({ seated: false, position: 1 });
});

test("someone who stops pinging drops out of the line", () => {
  let t = 0;
  const gate = createGate({ idleMs: 120_000, turnMs: 10 * 60_000, now: () => t });
  gate.claim("a@x.com");
  gate.claim("b@x.com");
  gate.claim("c@x.com");

  t = 60_000;
  gate.claim("a@x.com");
  gate.claim("c@x.com"); // b went quiet
  t = 121_000;
  expect(gate.claim("c@x.com")).toEqual({ seated: false, position: 1 }); // b's place is gone

  t = 240_000;
  gate.claim("c@x.com"); // a went quiet too
  expect(gate.claim("c@x.com")).toEqual({ seated: true, position: 0 });
});

test("an idle turn passes to the next person only when someone is waiting", () => {
  let t = 0;
  const gate = createGate({ idleMs: 10 * 60_000, turnMs: 5 * 60_000, now: () => t });
  gate.claim("a@x.com");
  t = 6 * 60_000;
  expect(gate.claim("a@x.com").seated).toBe(true); // nobody waiting: keep the seat

  gate.claim("b@x.com");
  expect(gate.claim("a@x.com")).toEqual({ seated: false, position: 1 }); // to the back, not out
  expect(gate.claim("b@x.com").seated).toBe(true);

  t += 4 * 60_000;
  expect(gate.allowChat("b@x.com")).toBe(true); // a chat restarts the turn timer
  t += 4 * 60_000;
  expect(gate.claim("b@x.com").seated).toBe(true);
});

test("each person gets 20 chats an hour and 100 a day", () => {
  let t = 0;
  const gate = createGate({ now: () => t });
  for (let i = 0; i < 20; i++) expect(gate.allowChat("a@x.com")).toBe(true);
  expect(gate.allowChat("a@x.com")).toBe(false);
  expect(gate.allowChat("b@x.com")).toBe(true); // limits are per person

  for (let hour = 1; hour <= 4; hour++) {
    t = hour * 3_600_000;
    for (let i = 0; i < 20; i++) expect(gate.allowChat("a@x.com")).toBe(true);
  }
  t = 5 * 3_600_000;
  expect(gate.allowChat("a@x.com")).toBe(false); // 100 today
  t = 24 * 3_600_000 + 1;
  expect(gate.allowChat("a@x.com")).toBe(true); // the first hour's chats have aged out
});

// ── Routes ──────────────────────────────────────────────────────────────────

function ollamaReply(content: string): Response {
  return new Response(
    `${JSON.stringify({ message: { content } })}\n${JSON.stringify({ message: { content: "" }, done: true })}\n`,
  );
}

function demoHandler(onChat: () => Response | Promise<Response> = () => ollamaReply("hello"), maxUsers = 1) {
  return createChatRequestHandler({
    googleClientId: CLIENT_ID,
    gate: { maxUsers },
    now: () => NOW,
    fetchImpl: certsFetch((url) => {
      if (url.endsWith("/api/tags")) return Response.json({ models: [] });
      if (url.endsWith("/api/chat")) return onChat();
      throw new Error(`Unexpected request: ${url}`);
    }),
  });
}

// Signs `email` in once per handler (through /api/login, like the page does)
// and reuses the session cookie.
const cookies = new WeakMap<object, Map<string, string>>();
async function sessionCookie(handler: ReturnType<typeof demoHandler>, email: string) {
  const jar = cookies.get(handler) ?? new Map<string, string>();
  cookies.set(handler, jar);
  if (!jar.has(email)) {
    const res = await handler(
      new Request("http://localhost/api/login", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ credential: await token({ email }) }),
      }),
    );
    expect(res.status).toBe(200);
    jar.set(email, res.headers.get("set-cookie")!.split(";")[0]!);
  }
  return jar.get(email)!;
}

async function post(handler: ReturnType<typeof demoHandler>, path: string, email?: string, body?: unknown) {
  const headers: Record<string, string> = { "content-type": "application/json" };
  if (email) headers.cookie = await sessionCookie(handler, email);
  return handler(
    new Request(`http://localhost${path}`, {
      method: "POST",
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
    }),
  );
}

const hello = { messages: [{ role: "user", content: "hello" }] };

test("signing in with a Google token sets a session cookie; a bad token gets a 401", async () => {
  const handler = demoHandler();
  const login = (credential: string) =>
    handler(
      new Request("http://localhost/api/login", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ credential }),
      }),
    );
  const ok = await login(await token({}));
  expect(ok.status).toBe(200);
  expect(await ok.json()).toEqual({ email: "visitor@example.com" });
  expect(ok.headers.get("set-cookie")).toMatch(/^session=.+; Path=\/; HttpOnly; Secure; SameSite=Lax/);
  expect((await login(await token({}, stranger.privateKey))).status).toBe(401);

  const config = await handler(new Request("http://localhost/api/config"));
  expect(await config.json()).toEqual({ googleClientId: CLIENT_ID, signInRequired: true });
});

test("signing out clears the cookie and gives up the place in line", async () => {
  const handler = demoHandler();
  await post(handler, "/api/session", "a@x.com");
  const out = await post(handler, "/api/logout", "a@x.com");
  expect(out.status).toBe(204);
  expect(out.headers.get("set-cookie")).toContain("Max-Age=0");
  expect(await (await post(handler, "/api/session", "b@x.com")).json()).toMatchObject({ seated: true });
});

test("chat and session routes need a signed-in visitor", async () => {
  const handler = demoHandler();
  expect((await post(handler, "/api/chat", undefined, hello)).status).toBe(401);
  expect((await post(handler, "/api/session")).status).toBe(401);

  const ok = await post(handler, "/api/chat", "a@x.com", hello);
  expect(ok.status).toBe(200);
  expect(await ok.text()).toContain("hello");
});

test("the session route gives queue numbers and the next person goes in on leave", async () => {
  const handler = demoHandler();
  expect(await (await post(handler, "/api/session", "a@x.com")).json()).toEqual({ seated: true, position: 0, email: "a@x.com" });
  expect(await (await post(handler, "/api/session", "b@x.com")).json()).toMatchObject({ seated: false, position: 1 });

  const early = await post(handler, "/api/chat", "b@x.com", hello);
  expect(early.status).toBe(503);
  expect(await early.json()).toMatchObject({ queued: true, position: 1 });

  expect((await post(handler, "/api/session/leave", "a@x.com")).status).toBe(204);
  expect(await (await post(handler, "/api/session", "b@x.com")).json()).toMatchObject({ seated: true, position: 0 });
  expect((await post(handler, "/api/chat", "b@x.com", hello)).status).toBe(200);
});

test("oversized chats get a 413 before reaching the model", async () => {
  let modelCalls = 0;
  const handler = demoHandler(() => {
    modelCalls++;
    return ollamaReply("hi");
  });
  const long = { messages: [{ role: "user", content: "x".repeat(MAX_USER_CHARS + 1) }] };
  const many = {
    messages: Array.from({ length: MAX_MESSAGES + 1 }, (_, i) => ({
      role: i % 2 ? "assistant" : "user",
      content: "hi",
    })),
  };
  const hugeReply = {
    messages: [
      { role: "user", content: "hi" },
      { role: "assistant", content: "y".repeat(5_000) },
      { role: "user", content: "and?" },
    ],
  };
  for (const body of [long, many, hugeReply]) {
    expect((await post(handler, "/api/chat", "a@x.com", body)).status).toBe(413);
  }
  expect(modelCalls).toBe(0);
});

test("the 21st chat in an hour gets a 429", async () => {
  const handler = demoHandler();
  for (let i = 0; i < 20; i++) {
    const res = await post(handler, "/api/chat", "a@x.com", hello);
    expect(res.status).toBe(200);
    await res.text();
  }
  expect((await post(handler, "/api/chat", "a@x.com", hello)).status).toBe(429);
});

test("chats run in parallel up to the seat count, and the next is told it is waiting", async () => {
  let finishHeld!: () => void;
  const held = new Promise<void>((resolve) => (finishHeld = resolve));
  let calls = 0;
  const handler = demoHandler(async () => {
    calls++;
    if (calls <= 2) await held;
    return ollamaReply(`answer ${calls}`);
  }, 2); // two seats: two chats at once

  const first = (await post(handler, "/api/chat", "a@x.com", hello)).text();
  const second = (await post(handler, "/api/chat", "b@x.com", hello)).text();
  await Bun.sleep(10);
  expect(calls).toBe(2); // both seats are at the model

  // A third chat (a second tab) waits for a slot.
  const third = (await post(handler, "/api/chat", "a@x.com", hello)).text();
  await Bun.sleep(10);
  expect(calls).toBe(2);
  finishHeld();

  expect(await first).not.toContain("waiting");
  expect(await second).not.toContain("waiting");
  const events = (await third).trim().split("\n").map((line) => JSON.parse(line));
  expect(events[0]).toEqual({ waiting: true });
  expect(events.some((e) => e.message?.content === "answer 3")).toBe(true);
});

test("without a Google client ID, local runs work with no sign-in", async () => {
  const handler = createChatRequestHandler({
    googleClientId: null,
    fetchImpl: async (input) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
      return url.endsWith("/api/tags") ? Response.json({ models: [] }) : ollamaReply("hi");
    },
  });
  expect((await post(handler as ReturnType<typeof demoHandler>, "/api/session")).status).toBe(200);
  expect((await post(handler as ReturnType<typeof demoHandler>, "/api/chat", undefined, hello)).status).toBe(200);
});

test("sign-ins and chats are logged with the visitor's email, never the question", async () => {
  const lines: string[] = [];
  const original = console.log;
  console.log = (line: unknown) => void lines.push(String(line));
  try {
    const handler = demoHandler();
    const res = await post(handler, "/api/chat", "a@x.com", { messages: [{ role: "user", content: "secret question" }] });
    await res.text();
  } finally {
    console.log = original;
  }
  const events = lines.flatMap((line) => {
    try {
      return [JSON.parse(line)];
    } catch {
      return [];
    }
  });
  expect(events).toContainEqual(expect.objectContaining({ event: "login", email: "a@x.com" }));
  expect(events).toContainEqual(expect.objectContaining({ event: "chat", email: "a@x.com", questionChars: 15 }));
  expect(lines.join("\n")).not.toContain("secret question");
});

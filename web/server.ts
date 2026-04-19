const server = Bun.serve({
    port:3000,
    fetch(req: Request): Response {
        const url = new URL(req.url);

    if (url.pathname === "/") {
      return new Response("Hello from Bun! 🐇");
    }

    if (url.pathname === "/json") {
      return Response.json({ message: "Hello", ts: Date.now() });
    }

    return new Response("Not Found", { status: 404 });
    },

});
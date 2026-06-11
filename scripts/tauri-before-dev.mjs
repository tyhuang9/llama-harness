import { spawn } from "node:child_process";
import net from "node:net";

const children = [];

function run(name, command, args, options = {}) {
  const child = spawn(command, args, {
    cwd: new URL("..", import.meta.url),
    env: process.env,
    stdio: ["ignore", "pipe", "pipe"],
    ...options,
  });

  children.push(child);

  child.stdout.on("data", (chunk) => {
    process.stdout.write(`[${name}] ${chunk}`);
  });
  child.stderr.on("data", (chunk) => {
    process.stderr.write(`[${name}] ${chunk}`);
  });
  child.on("exit", (code, signal) => {
    if (!shuttingDown) {
      console.error(`[${name}] exited with ${signal || code}`);
      shutdown(code || 1);
    }
  });
}

let shuttingDown = false;

function shutdown(code = 0) {
  shuttingDown = true;
  for (const child of children) {
    if (!child.killed) {
      child.kill("SIGTERM");
    }
  }
  setTimeout(() => process.exit(code), 250);
}

process.on("SIGINT", () => shutdown(0));
process.on("SIGTERM", () => shutdown(0));

function portIsOpen(port) {
  return new Promise((resolve) => {
    const socket = net.connect({ host: "127.0.0.1", port });
    socket.once("connect", () => {
      socket.end();
      resolve(true);
    });
    socket.once("error", () => resolve(false));
    socket.setTimeout(500, () => {
      socket.destroy();
      resolve(false);
    });
  });
}

if (await portIsOpen(8787)) {
  console.log("[api] reusing existing server on http://127.0.0.1:8787");
} else {
  run("api", "cargo", ["run", "--manifest-path", "server/Cargo.toml"]);
}

if (await portIsOpen(5174)) {
  console.log("[ui] reusing existing dev server on http://127.0.0.1:5174");
} else {
  run("ui", "npm", ["--prefix", "admin-ui", "run", "dev"]);
}

setInterval(() => undefined, 1_000);

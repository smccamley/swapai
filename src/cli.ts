#!/usr/bin/env node
import { startClassifiersUi } from "./classifiers-ui.js";

const args = process.argv.slice(2);
const command = args.shift();

if (command === "--help" || command === "-h" || command === undefined) {
  printHelp();
  process.exit(command === undefined ? 1 : 0);
}
if (command !== "classifiers-ui") {
  console.error(`Unknown command: ${command}`);
  printHelp();
  process.exit(1);
}

const options = parseOptions(args);
const server = await startClassifiersUi(options);
console.log(`SwapAI classifier monitor: ${server.url}`);
console.log(`Data directory: ${options.dataDirectory ?? process.env.SWAPAI_DATA_DIRECTORY ?? ".swapai"}`);

for (const signal of ["SIGINT", "SIGTERM"] as const) {
  process.once(signal, () => {
    void server.close().finally(() => process.exit(0));
  });
}

function parseOptions(values: string[]): {
  dataDirectory?: string;
  host?: string;
  port?: number;
} {
  const options: { dataDirectory?: string; host?: string; port?: number } = {};
  for (let index = 0; index < values.length; index += 1) {
    const flag = values[index];
    if (flag === "--help" || flag === "-h") {
      printHelp();
      process.exit(0);
    }
    const value = values[index + 1];
    if (value === undefined) throw new TypeError(`${flag} needs a value`);
    if (flag === "--data-directory") options.dataDirectory = value;
    else if (flag === "--host") options.host = value;
    else if (flag === "--port") options.port = Number(value);
    else throw new TypeError(`Unknown option: ${flag}`);
    index += 1;
  }
  return options;
}

function printHelp(): void {
  console.log(`Usage: swapai classifiers-ui [options]

Starts the local SwapAI classifier monitor.

Options:
  --data-directory <path>  SwapAI data directory (default: .swapai)
  --host <host>            Listen host (default: 127.0.0.1)
  --port <port>            Listen port (default: 4789)
  -h, --help               Show this help`);
}

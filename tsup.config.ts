import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/index.ts", "src/effect.ts", "src/classifiers-ui.ts", "src/runpod.ts", "src/cli.ts"],
  format: ["esm"],
  platform: "node",
  target: "node22",
  dts: true,
  sourcemap: true,
  clean: true,
  external: ["effect", "node:sqlite"],
});

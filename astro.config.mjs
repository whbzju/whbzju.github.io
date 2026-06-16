import { defineConfig } from "astro/config";

export default defineConfig({
  site: "https://wujia.me",
  output: "static",
  publicDir: "./site-public",
  outDir: "./dist",
});

import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

// Unit and component tests. JSX is compiled by Vite's built-in transform using tsconfig's `"jsx": "react-jsx"`,
// so no React plugin is needed (there is no dev server here, only tests).
export default defineConfig({
  resolve: {
    alias: [{ find: /^@\//, replacement: fileURLToPath(new URL("./", import.meta.url)) }],
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./vitest.setup.ts"],
    include: ["__tests__/**/*.test.{ts,tsx}"],
    css: false,
    restoreMocks: true,
    unstubGlobals: true,
  },
});

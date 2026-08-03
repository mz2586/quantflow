/**
 * ESLint flat config.
 *
 * Type-aware on purpose: the rules that matter here — no floating promises, no unsafe
 * `any` reaching a render — cannot be checked without type information, and those are
 * exactly the mistakes that put a wrong number in front of an operator.
 */

import js from "@eslint/js";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";
import tseslint from "typescript-eslint";

export default tseslint.config(
  { ignores: ["dist", "coverage", "node_modules"] },
  js.configs.recommended,
  // Type-aware rules apply to sources only. The build's own config files (postcss,
  // tailwind, eslint) are plain JS outside the tsconfig, and linting them with rules
  // that demand type information crashes the run rather than reporting anything.
  ...tseslint.configs.strictTypeChecked.map((config) => ({
    ...config,
    files: ["**/*.{ts,tsx}"],
  })),
  ...tseslint.configs.stylisticTypeChecked.map((config) => ({
    ...config,
    files: ["**/*.{ts,tsx}"],
  })),
  {
    files: ["**/*.{ts,tsx}"],
    languageOptions: {
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
      globals: {
        window: "readonly",
        document: "readonly",
        WebSocket: "readonly",
        fetch: "readonly",
        console: "readonly",
      },
    },
    plugins: {
      "react-hooks": reactHooks,
      "react-refresh": reactRefresh,
    },
    rules: {
      ...reactHooks.configs.recommended.rules,
      "react-refresh/only-export-components": ["warn", { allowConstantExport: true }],
      // A dropped promise in a dashboard means a panel that silently stops updating.
      "@typescript-eslint/no-floating-promises": "error",
      "@typescript-eslint/no-misused-promises": "error",
      // Money is a string end to end; an implicit coercion is how it gets corrupted.
      "@typescript-eslint/restrict-template-expressions": [
        "error",
        { allowNumber: true },
      ],
    },
  },
  {
    files: ["**/*.test.{ts,tsx}"],
    rules: { "@typescript-eslint/no-non-null-assertion": "off" },
  },
  {
    files: ["**/*.js", "**/*.cjs"],
    languageOptions: { globals: { process: "readonly", module: "writable" } },
  },
);

# TypeScript Reference

A comprehensive reference written for someone with a strong Python + general engineering background, who needs to read and write TypeScript fluently for backend (Bun) and frontend (vanilla DOM) work. Not a tutorial — a reference. Read it once cover-to-cover, then come back when stuck.

Roughly ordered by "what you'll see first" and "what you'll need most often" rather than by language theory.

---

## 1. The mental model

TypeScript = JavaScript + a static type-checker.

The browser/runtime never sees TypeScript. A tool (Bun, `tsc`, esbuild, swc) strips the type annotations and emits plain JavaScript. The types exist only at compile time, to catch mistakes before runtime.

**Implication 1:** Every TS program is also a valid JS program if you delete the type annotations. When stuck, mentally erase the types and ask "would this JavaScript work?"

**Implication 2:** Types do not exist at runtime. You cannot write `if (x is Message)` at runtime. You write *type guards* (functions that look at the value's shape and tell TS what type it must be) instead.

**Implication 3:** TypeScript uses **structural typing**, not nominal typing. Two types are compatible if they have the same shape, regardless of name. This is different from Java/C# but similar to Python's duck typing — if it walks like a duck, TS treats it like a duck.

```ts
interface Named { name: string; }
const obj = { name: "Alice", age: 30 };
const n: Named = obj;  // works — obj has a name field, that's enough
```

**Implication 4:** TypeScript is *unsound* on purpose. You can lie to it (with `as`, `any`, etc.) and it will believe you. The runtime won't. Lies bite back.

---

## 2. The basic types

| Type | Example values |
|------|----------------|
| `string` | `"hello"`, backtick template strings |
| `number` | `42`, `3.14`, `NaN`, `Infinity` (no separate int/float) |
| `boolean` | `true`, `false` |
| `bigint` | `42n` (rarely needed) |
| `symbol` | `Symbol("x")` (rarely needed) |
| `null` | `null` |
| `undefined` | `undefined` |
| `void` | function returns nothing |
| `never` | function never returns (throws or infinite loops) |
| `unknown` | "I don't know what this is yet" — safe |
| `any` | "turn off type checking for this value" — escape hatch, avoid |

`null` vs `undefined`: JS has both, and they're different values. With `strictNullChecks` on (which you should always have), TS treats them as distinct from other types. A function that might return nothing is `T | undefined` or `T | null` depending on what it actually returns.

`unknown` vs `any`: both can hold anything, but `unknown` forces you to narrow before using it (you can't call methods on `unknown` until you've checked what it is). Always prefer `unknown` over `any`.

---

## 3. Variables and inference

```ts
let x: number = 5;       // explicit type
let y = 5;               // inferred as number
const z = 5;             // inferred as the literal type 5 (because const)
let a: number;           // declared but uninitialized — type number, value undefined
```

**Inference is your friend.** Annotate types when:
- The signature is a public API (function arguments, return types)
- Inference picks something too narrow or too wide
- You want documentation

Don't annotate when inference is obvious. `let count: number = 0` is noise; `let count = 0` is fine.

`const` narrows the type. `const x = "hello"` has type `"hello"` (a literal type), not `string`. This matters for unions:

```ts
const mode = "finetune";  // type: "finetune"
let modeLet = "finetune"; // type: string
```

---

## 4. Functions

```ts
function greet(name: string): string {
  return `Hello, ${name}`;
}

const greet2 = (name: string): string => `Hello, ${name}`;

// Optional parameter
function greet3(name: string, greeting?: string): string {
  return `${greeting ?? "Hello"}, ${name}`;
}

// Default parameter
function greet4(name: string, greeting = "Hello"): string {
  return `${greeting}, ${name}`;
}

// Rest parameters
function sum(...nums: number[]): number {
  return nums.reduce((a, b) => a + b, 0);
}
```

**Function type expression:**
```ts
type GreetFn = (name: string) => string;
const greeter: GreetFn = (name) => `Hi ${name}`;
```

The `=>` here means "function returning"; it's syntactic, not runtime behavior.

**Return type inference works.** You usually don't need `: string` on the return; TS infers it. But on **public functions and module exports, annotate the return type** — it catches bugs and makes the API explicit.

`void` means "I don't care about the return value" — slightly different from "returns undefined":
```ts
type Callback = () => void;
const cb: Callback = () => 42;  // OK, void means caller ignores the return
```

---

## 5. Object types: `interface` vs `type`

Both define object shapes. They overlap heavily. Use either.

```ts
interface User {
  id: number;
  name: string;
  email?: string;       // optional
  readonly createdAt: Date;  // can't reassign after creation
}

type User2 = {
  id: number;
  name: string;
  email?: string;
  readonly createdAt: Date;
};
```

**Differences worth knowing:**

- `interface` can be extended via `extends`; `type` is composed via `&` (intersection).
- `interface` declarations with the same name **merge** (declaration merging). `type` aliases collide.
- `type` can express things `interface` can't: unions, primitives, mapped types, conditional types.

**Rule of thumb:** Use `interface` for shapes meant to be extended. Use `type` for unions, mapped types, primitives, and one-off shapes. In practice, just pick one and be consistent within a file.

```ts
interface Animal { name: string; }
interface Dog extends Animal { breed: string; }

type Animal2 = { name: string; };
type Dog2 = Animal2 & { breed: string; };
```

**Index signatures** for "an object with arbitrary string keys mapping to T":
```ts
interface StringMap {
  [key: string]: string;
}
const m: StringMap = { foo: "bar", baz: "qux" };
```

---

## 6. Arrays and tuples

```ts
const xs: number[] = [1, 2, 3];
const ys: Array<number> = [1, 2, 3];  // identical
const mixed: (string | number)[] = ["a", 1, "b"];

// Tuple — fixed-length, position-typed
const pair: [string, number] = ["age", 30];
const named: [name: string, age: number] = ["Alice", 30];  // labels for docs

// Readonly
const frozen: readonly number[] = [1, 2, 3];
frozen.push(4);  // error
```

You'll see tuples returned from React-style hooks (`const [count, setCount] = useState(0)`) and from utility libraries. In your build they show up rarely.

---

## 7. Union types

The single most useful TypeScript feature. A value can be one of several types.

```ts
type Mode = "finetune" | "rag";        // string literal union
type Id = string | number;              // primitive union
type Result = Success | Failure;        // object union
```

You can't access a property unless it exists on **all** members of the union:

```ts
type Cat = { meow: () => void };
type Dog = { bark: () => void };
type Pet = Cat | Dog;

function noise(p: Pet) {
  p.meow();  // error — Dog doesn't have meow
  if ("meow" in p) p.meow();  // OK — narrowed to Cat
}
```

This narrowing pattern is the workhorse of TS day-to-day code. See section 12 for more.

---

## 8. Intersection types

`&` combines types — the value must satisfy *all* of them:

```ts
type WithId = { id: string };
type WithTimestamp = { createdAt: Date };
type Record = WithId & WithTimestamp;
// Record requires both id and createdAt
```

Used heavily in extending types and combining mixins. Less common day-to-day than unions.

---

## 9. Literal types

A literal type is a type that contains exactly one value:

```ts
type Yes = "yes";
const y: Yes = "yes";
const n: Yes = "no";  // error
```

Mostly useful in unions:

```ts
type Direction = "up" | "down" | "left" | "right";
type HttpMethod = "GET" | "POST" | "PUT" | "DELETE";
type StatusCode = 200 | 201 | 400 | 404 | 500;
```

This pattern replaces enums in modern TS. Pick it; enums are largely an anti-pattern in 2026.

---

## 10. `null`, `undefined`, optional, nullable

With `strictNullChecks` on (always have it on):

- `T` does NOT include `null` or `undefined`
- `T | null`, `T | undefined`, or `T | null | undefined` includes them explicitly
- Optional fields (`field?: T`) have type `T | undefined`
- Optional parameters (`param?: T`) have type `T | undefined`

```ts
interface User {
  name: string;
  age?: number;  // age is number | undefined
}

const u: User = { name: "A" };
u.age + 1;        // error — possibly undefined
u.age ?? 0;       // OK — nullish coalescing, falls back to 0
u.age?.toFixed(); // OK — optional chaining
```

**`??` (nullish coalescing):** returns the right side only if the left is `null` or `undefined`. Different from `||` (which also falls through on falsy values like `0`, `""`, `false`).

```ts
const a = 0 || 10;   // 10  (because 0 is falsy)
const b = 0 ?? 10;   // 0   (because 0 is not null/undefined)
```

**`?.` (optional chaining):** returns `undefined` if the LHS is `null`/`undefined`, otherwise accesses the property.

```ts
user?.name           // undefined if user is null/undefined
user?.profile?.bio   // chain
arr?.[0]             // optional indexed access
fn?.()               // optional call
```

**`!` (non-null assertion):** "trust me, this is not null/undefined." Compiles away. If you're wrong, runtime breaks. Use sparingly.

```ts
const el = document.getElementById("foo")!;  // I know it exists
```

---

## 11. Type narrowing

The runtime check that lets TS update its understanding of a value's type.

```ts
function f(x: string | number) {
  if (typeof x === "string") {
    x.toUpperCase();  // x is string here
  } else {
    x.toFixed();      // x is number here
  }
}
```

The narrowing operators TS understands:

- `typeof x === "string" | "number" | "boolean" | "object" | "function" | "undefined" | "symbol" | "bigint"`
- `x instanceof Class` — narrows to the class
- `"key" in obj` — narrows to types that have that key
- Equality checks against literal types: `if (x === "rag")`
- Truthiness: `if (x)` removes `null`, `undefined`, `0`, `""`, `false`
- `Array.isArray(x)`

**User-defined type guards** (functions that return `x is T`):

```ts
interface Cat { meow: () => void; }
function isCat(x: unknown): x is Cat {
  return typeof x === "object" && x !== null && "meow" in x;
}

const pet: unknown = ...;
if (isCat(pet)) {
  pet.meow();  // narrowed to Cat
}
```

These are how you turn `unknown` (e.g., parsed JSON) into a real type safely.

**Discriminated unions** — the most powerful narrowing pattern:

```ts
type Shape =
  | { kind: "circle"; radius: number }
  | { kind: "square"; side: number }
  | { kind: "rectangle"; width: number; height: number };

function area(s: Shape): number {
  switch (s.kind) {
    case "circle":    return Math.PI * s.radius ** 2;
    case "square":    return s.side ** 2;
    case "rectangle": return s.width * s.height;
  }
}
```

The `kind` field is a *discriminant* — TS narrows `s` based on its value. Use this pattern for events, results, states, anything with variants.

---

## 12. Generics

Generic = "a function or type parameterized by another type."

```ts
function identity<T>(x: T): T {
  return x;
}

const a = identity<string>("hello");  // T = string
const b = identity(42);                // T inferred as number
```

The `<T>` declares a type parameter. `T` is just a name — by convention single uppercase letters (`T`, `U`, `K`, `V`) for simple cases, descriptive names (`TItem`, `TKey`) for complex ones.

Common in containers and utilities:

```ts
function first<T>(arr: T[]): T | undefined {
  return arr[0];
}

interface ApiResponse<T> {
  data: T;
  error: string | null;
}

const userResp: ApiResponse<User> = { data: user, error: null };
```

**Constraints** (`extends`) — limit what types are allowed:

```ts
function getKey<T extends object, K extends keyof T>(obj: T, key: K): T[K] {
  return obj[key];
}
```

`keyof T` is the union of `T`'s keys as string literal types. So `keyof User` would be `"id" | "name" | "age"`.

**Default type parameters:**
```ts
interface Box<T = string> { value: T; }
const b: Box = { value: "x" };  // T defaults to string
```

You'll write generics rarely. You'll *read* them constantly.

---

## 13. Promises and async/await

Same model as Python, slightly different syntax.

```ts
async function fetchUser(id: string): Promise<User> {
  const response = await fetch(`/users/${id}`);
  return await response.json();  // type Promise<any> — see below
}
```

`async` functions always return `Promise<T>`. `await` unwraps a `Promise<T>` to a `T`.

**Important gotcha:** `response.json()` returns `Promise<any>` because TS can't know the shape of arbitrary JSON. You should narrow:

```ts
async function fetchUser(id: string): Promise<User> {
  const response = await fetch(`/users/${id}`);
  const data: unknown = await response.json();
  if (!isUser(data)) throw new Error("invalid user");
  return data;
}
```

In practice, many codebases skip this and trust the API contract. That's pragmatic but unsafe — bad data crashes at runtime far from where it entered.

**Parallel awaits:**
```ts
const [a, b] = await Promise.all([fetchA(), fetchB()]);
```

**Async iterators** — what streaming uses:
```ts
async function* tokens(): AsyncGenerator<string> {
  for (const t of someStream) {
    yield t;
  }
}

for await (const token of tokens()) {
  console.log(token);
}
```

`async function*` is a generator that yields asynchronously. `for await` consumes it. This is the heart of streaming APIs in your chat app.

---

## 14. Modules: `import` and `export`

TS uses ES modules. Each `.ts` file is a module.

```ts
// types.ts
export interface Message { role: string; content: string; }
export type Mode = "finetune" | "rag";
export const DEFAULT_MODE: Mode = "rag";

// chat.ts
import { Message, Mode, DEFAULT_MODE } from "./types";
import type { Message as Msg } from "./types";  // type-only import
import * as types from "./types";                // import everything as namespace
```

**`import type` vs `import`:** `import type` only imports the type (compiled away to nothing). `import` imports the runtime value. Use `import type` for things that are types-only — interfaces, type aliases, literal unions. It's slightly more efficient and clearer to readers.

**Default exports:** one per file, no name needed at import.
```ts
// thing.ts
export default function thing() { ... }

// other.ts
import thing from "./thing";  // any name works
```

Avoid default exports unless a library forces them. Named exports are easier to refactor and grep.

**Re-exports:**
```ts
export { Message } from "./types";       // re-export by name
export * from "./types";                  // re-export everything
```

**Path resolution:** `"./types"` — relative. `"react"` — package. `"~/foo"` or `"@/foo"` — alias configured in `tsconfig.json` (`paths`).

---

## 15. Classes

You'll use these less in modern TS code than you might expect. They exist; here's the gist:

```ts
class Counter {
  private count = 0;

  constructor(initial: number = 0) {
    this.count = initial;
  }

  increment(): void {
    this.count++;
  }

  get value(): number {
    return this.count;
  }
}

const c = new Counter(10);
c.increment();
console.log(c.value);
```

**Access modifiers:** `public` (default), `private`, `protected`. These are *compile-time only* — at runtime, all fields are accessible.

**Modern alternative — `#` private fields:** actually private at runtime.
```ts
class Counter {
  #count = 0;
  increment() { this.#count++; }
}
```

**Parameter properties** — shorthand for "store this constructor arg as a field":
```ts
class User {
  constructor(public name: string, private age: number) {}
}
// equivalent to declaring and assigning name and age in the constructor body
```

For your chat app, you probably won't need classes at all. Prefer plain objects + functions.

---

## 16. Utility types (the ones worth knowing)

Built-in helpers in the standard library. The most useful:

```ts
// Make all fields optional
Partial<User>  // { id?: string; name?: string; email?: string }

// Make all fields required (removes ?)
Required<User>

// Make all fields readonly
Readonly<User>

// Pick a subset of fields
Pick<User, "id" | "name">  // { id: string; name: string }

// Omit specific fields
Omit<User, "email">

// Object type with keys K and values V
Record<string, number>  // { [key: string]: number }

// The keys of T as a union of literal types
keyof User  // "id" | "name" | "email"

// The return type of a function
ReturnType<typeof myFn>

// The argument types of a function
Parameters<typeof myFn>  // tuple type

// Awaited<Promise<T>> = T (unwraps a Promise type)
Awaited<Promise<string>>  // string

// Make a type non-nullable
NonNullable<string | null | undefined>  // string
```

`typeof` here is the TypeScript `typeof` operator, *not* the JS one — it gets the type of a value.

```ts
const config = { host: "localhost", port: 3000 };
type Config = typeof config;  // { host: string; port: number }
```

This is enormously useful for inferring types from values you've already written.

---

## 17. Type assertions and casting

```ts
const x = someValue as string;       // "trust me, this is a string"
const y = <string>someValue;          // older syntax, avoid in TSX/JSX contexts
```

`as` is an unchecked assertion at compile time. Runtime is unaffected. If you assert wrong, the code may break later in surprising places. Use sparingly:

- DOM queries (`as HTMLTextAreaElement`)
- Parsing untrusted JSON (after validation, narrow via type guards instead)
- When you *truly* know more than TS does

**`as const`** — special: makes a value immutable and narrows everything to literal types:

```ts
const point = { x: 1, y: 2 };          // type: { x: number; y: number }
const point2 = { x: 1, y: 2 } as const; // type: { readonly x: 1; readonly y: 2 }

const arr = ["a", "b"] as const;        // type: readonly ["a", "b"]
```

Useful when you want to derive a union from a list:

```ts
const MODES = ["finetune", "rag"] as const;
type Mode = typeof MODES[number];  // "finetune" | "rag"
```

**`satisfies`** — newer, often what you actually want instead of `as`:

```ts
const config = {
  host: "localhost",
  port: 3000,
} satisfies Config;
```

`satisfies` checks the value matches the type *without* widening it. The variable still has the literal-narrowed type. With `as`, you lose narrowing. Use `satisfies` when assigning a literal value that should match a type.

---

## 18. DOM types

Important for frontend work. The DOM types come from `lib: ["DOM"]` in `tsconfig.json`.

The relevant hierarchy:
- `Node` — anything in the DOM tree
- `Element` — an HTML/SVG/MathML element
- `HTMLElement` — an HTML element
- `HTMLInputElement`, `HTMLButtonElement`, `HTMLDivElement`, `HTMLSelectElement`, `HTMLTextAreaElement`, etc. — specific tags

```ts
// Generic — returns HTMLElement | null
const el = document.getElementById("foo");

// Cast to specific type
const input = document.getElementById("foo") as HTMLInputElement;
input.value;  // OK now

// Or use querySelector with a generic
const btn = document.querySelector<HTMLButtonElement>("#submit");
btn?.click();
```

`querySelector` returns `Element | null` by default; using the generic version `querySelector<HTMLButtonElement>` returns `HTMLButtonElement | null` directly without an assertion.

**Event types:**
```ts
button.addEventListener("click", (e: MouseEvent) => {
  e.preventDefault();
});

input.addEventListener("input", (e: Event) => {
  const target = e.target as HTMLInputElement;
  console.log(target.value);
});
```

Most DOM event listener type inference works automatically. If TS can't tell the event type, fall back to `Event` and cast `e.target`.

---

## 19. Common error patterns

TypeScript has no built-in `Result` type, no checked exceptions. Errors are thrown like in JS.

```ts
try {
  const data = await fetchData();
} catch (err) {
  // err is unknown by default in modern TS
  if (err instanceof Error) {
    console.error(err.message);
  } else {
    console.error("unknown error", err);
  }
}
```

**Patterns to know:**

1. **Throw and catch** — fine for unexpected failures.
2. **Return a discriminated union** — for expected failures:
   ```ts
   type Result<T, E> =
     | { ok: true; value: T }
     | { ok: false; error: E };
   ```
3. **Throw and let it propagate** — for "this should never happen, fail loudly."

The discriminated-union approach is gaining popularity in Rust-influenced TS codebases. Pure throw-based is still the JS norm. Pick one per project; don't mix.

---

## 20. The gotchas list

The things that will bite you, ranked by how often.

**1. `any` is contagious.** One `any` poisons every value derived from it. Avoid. Prefer `unknown` and narrow.

**2. Object property access on union types fails until narrowed.**
```ts
type X = { a: number } | { b: string };
const x: X = ...;
x.a;  // error
```

**3. `Array.prototype.find` returns `T | undefined`, not `T`.** So does `arr[i]` when `noUncheckedIndexedAccess` is on. Always handle the `undefined` case.

**4. `JSON.parse` returns `any`.** Treat parsed JSON as `unknown` and narrow:
```ts
const data: unknown = JSON.parse(text);
if (typeof data === "object" && data !== null && "name" in data) { ... }
```

**5. Mutability is not tracked.** `const x = { a: 1 }` doesn't prevent `x.a = 2`. Use `readonly` fields if mutation matters.

**6. Variance traps with arrays.** `T[]` is *covariant* — `string[]` is assignable to `(string | number)[]`. This is unsound but pragmatic. Mostly fine to ignore.

**7. Excess property checks on object literals.** Pass an inline object with extra keys to a typed parameter, you get an error. Pass the same object via a variable, you don't:
```ts
interface Opts { name: string; }
function f(o: Opts) {}
f({ name: "x", extra: 1 });  // error
const o = { name: "x", extra: 1 };
f(o);  // OK — TS only checks structural compatibility
```
This is intentional — catching typos in object literals.

**8. `this` types and arrow functions.** Inside a regular function, `this` is dynamic. Inside an arrow, `this` is lexical. In class methods, prefer arrow functions for callbacks to keep `this` bound.

**9. Type assertions can lie.** `unknown as User` will compile and crash at runtime if the value isn't a User. Validate at boundaries.

**10. `void` is weird in callbacks.** A function returning `T` can be assigned to a `() => void` callback type. The callback's return value is just ignored. This is by design but surprises people.

---

## 21. `tsconfig.json` — the settings that matter

Most settings have defaults that don't matter. The ones that matter:

```jsonc
{
  "compilerOptions": {
    // Output target
    "target": "ES2022",         // what JS to emit; ES2022 = modern engines
    "module": "ESNext",         // emit ES modules
    "moduleResolution": "Bundler", // for Vite/esbuild/Bun; "Node16" for Node

    // Strictness — TURN ON
    "strict": true,             // enables all strict checks below
    // strict implies all of these:
    // "noImplicitAny": true,
    // "strictNullChecks": true,
    // "strictFunctionTypes": true,
    // "strictPropertyInitialization": true,
    // "alwaysStrict": true,

    // Extra strictness worth enabling
    "noUncheckedIndexedAccess": true,  // arr[i] is T | undefined
    "noImplicitOverride": true,
    "noFallthroughCasesInSwitch": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,

    // What APIs are available
    "lib": ["ES2022", "DOM", "DOM.Iterable"],  // DOM if frontend
    "types": ["bun-types"],                     // for Bun

    // How files relate
    "esModuleInterop": true,    // play nicely with CJS imports
    "skipLibCheck": true,       // don't typecheck node_modules
    "forceConsistentCasingInFileNames": true,
    "isolatedModules": true,    // each file independently transpilable

    // Output
    "outDir": "dist",
    "noEmit": false,
    "sourceMap": true,
    "declaration": true,        // emit .d.ts files (for libraries)
    "declarationMap": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}
```

`strict: true` is non-negotiable for a new project. Everything beyond that is taste.

---

## 22. What you can punt on (for now)

These exist, you'll see them, you don't need to write them. Skip until they bite you:

- **Conditional types:** `T extends U ? X : Y`
- **Mapped types:** `{ [K in keyof T]: ... }`
- **Template literal types:** `` `prefix-${string}` ``
- **`infer` keyword** in conditional types
- **Decorators**
- **Namespaces** (legacy — use modules)
- **Triple-slash directives** (legacy)
- **Module augmentation / declaration merging** beyond simple cases

These show up in advanced library code. Read them when you encounter them. Don't try to learn them upfront.

---

## 23. The minimum viable TypeScript mental checklist

When something looks weird, run through this list:

1. What does this look like with all the type annotations deleted? Is *that* JS valid?
2. Is this a value or a type? (TS has two namespaces. `User` can mean both depending on context.)
3. What's the inferred type? (Hover in the editor.)
4. Is `null` or `undefined` allowed here?
5. Am I narrowed? Can TS know what type this is at this point?
6. Is this an `as` lying about something?

90% of "what is going on" questions resolve with one of these.

---

## 24. Ecosystem notes (read once)

- **Type definitions for npm packages** are usually shipped with the package (`types` field in package.json) or in `@types/<pkg>` (DefinitelyTyped). `bun add foo` installs both if `@types/foo` exists.
- **Editor support:** VS Code and Neovim (with `tsserver` or `typescript-language-server`) give you hover-to-see-types, jump-to-definition, autocomplete, inline errors. Use it. Reading TS without an editor that shows inferred types is a mistake.
- **`tsc --noEmit`** type-checks without producing output. Useful in CI and as a pre-commit step.
- **Bun, esbuild, swc, tsc:** all transpile TS to JS. Bun and esbuild are *transpilers* — they strip types but don't type-check. `tsc` does both. In practice, you transpile with Bun (fast) and type-check with `tsc --noEmit` separately.

---

## End

You don't need more. The 80% you'll write daily is in sections 2-14. The 18% you'll read but rarely write is in 15-20. The 2% lives in section 22 and can be looked up when you encounter it.

If you finish reading this and feel a bit overwhelmed, that's normal — you don't internalize a language by reading. You internalize by writing 200 lines of it. Section 22 is permission to ignore everything that didn't fit in your head; the build plan is where the actual learning happens.

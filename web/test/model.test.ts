import { expect, test } from "bun:test";

import { pickLatestModel } from "../src/app";

test("pickLatestModel returns the newest matching model", () => {
  const model = pickLatestModel(
    [
      { name: "other:1", modified_at: "2026-04-18T10:00:00Z" },
      { name: "chembl-drug-chat:1b", modified_at: "2026-04-18T10:00:00Z" },
      { name: "chembl-drug-chat:latest", modified_at: "2026-04-19T10:00:00Z" },
    ],
    "chembl-drug-chat",
  );

  expect(model).toBe("chembl-drug-chat:latest");
});

test("pickLatestModel returns null when nothing matches", () => {
  expect(
    pickLatestModel([{ name: "other:1", modified_at: "2026-04-19T10:00:00Z" }], "chembl-drug-chat"),
  ).toBeNull();
});

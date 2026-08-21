import { describe, expect, it } from "vitest";
import { parseSseBuffer, parseSseData } from "./sse";

describe("parseSseBuffer", () => {
  it("sépare un événement complet et garde le reste incomplet", () => {
    const { events, remainder } = parseSseBuffer('data: {"a":1}\n\ndata: {"b"');
    expect(events).toEqual(['data: {"a":1}']);
    expect(remainder).toBe('data: {"b"');
  });

  it("extrait plusieurs événements complets d'un seul coup", () => {
    const { events, remainder } = parseSseBuffer('data: {"a":1}\n\ndata: {"b":2}\n\n');
    expect(events).toEqual(['data: {"a":1}', 'data: {"b":2}']);
    expect(remainder).toBe("");
  });

  it("retourne tout en remainder quand aucun événement n'est complet", () => {
    const { events, remainder } = parseSseBuffer('data: {"a"');
    expect(events).toEqual([]);
    expect(remainder).toBe('data: {"a"');
  });
});

describe("parseSseData", () => {
  it("extrait le JSON d'une ligne data:", () => {
    expect(parseSseData<{ status: string }>('data: {"status":"running"}')).toEqual({ status: "running" });
  });

  it("retourne null pour un événement sans ligne data:", () => {
    expect(parseSseData("event: error")).toBeNull();
  });

  it("retourne null pour un JSON malformé, sans lever d'exception", () => {
    expect(parseSseData("data: {invalide")).toBeNull();
  });
});

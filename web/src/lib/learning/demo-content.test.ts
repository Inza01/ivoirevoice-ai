import { describe, expect, it } from "vitest";

import {
  DEMO_COURSES,
  DEMO_LESSONS,
  getDemoCourse,
  getDemoLesson,
} from "@/lib/learning/demo-content";

describe("demo learning structures", () => {
  it("resolves only known public demo identifiers", () => {
    expect(getDemoCourse("premiers-pas")).toBe(DEMO_COURSES[0]);
    expect(getDemoLesson("dire-bonjour")).toBe(DEMO_LESSONS[0]);
    expect(getDemoCourse("../../private")).toBeUndefined();
    expect(getDemoLesson("unknown")).toBeUndefined();
  });

  it("marks linguistic modules as structures or content requiring review", () => {
    for (const course of DEMO_COURSES) {
      for (const moduleTitle of course.modules) {
        expect(moduleTitle.fr).toMatch(/structure|à valider/i);
        expect(moduleTitle.en).toMatch(/structure|to review/i);
      }
    }
  });
});

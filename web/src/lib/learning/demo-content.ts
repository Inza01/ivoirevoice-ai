import type { UiLocale } from "@/lib/languages/registry";

type LocalizedText = Readonly<Record<UiLocale, string>>;

export const DEMO_COURSES = [
  {
    id: "premiers-pas",
    title: { fr: "Premiers pas en Dioula", en: "First steps in Dioula" },
    description: {
      fr: "Structure de démonstration pour découvrir les bases et le parcours.",
      en: "A demonstration structure for exploring the basics and the learning path.",
    },
    level: { fr: "Débutant · A1 interne", en: "Beginner · internal A1" },
    modules: [
      { fr: "Comprendre le parcours — structure", en: "Understanding the path — structure" },
      { fr: "Salutations — contenu à valider", en: "Greetings — content to review" },
      { fr: "Vie quotidienne — contenu à valider", en: "Daily life — content to review" },
    ],
  },
  {
    id: "vie-quotidienne",
    title: { fr: "Vie quotidienne", en: "Daily life" },
    description: {
      fr: "Structure de cours autour des situations courantes et du vocabulaire.",
      en: "A course structure focused on everyday situations and vocabulary.",
    },
    level: { fr: "Débutant avancé · A2 interne", en: "Advanced beginner · internal A2" },
    modules: [
      { fr: "Famille — contenu à valider", en: "Family — content to review" },
      { fr: "Marché et commerce — contenu à valider", en: "Market — content to review" },
      { fr: "Voyage — contenu à valider", en: "Travel — content to review" },
    ],
  },
  {
    id: "ecoute-pratique",
    title: { fr: "Écoute et pratique", en: "Listening and practice" },
    description: {
      fr: "Future progression mêlant écoute, compréhension et transcription.",
      en: "A future path combining listening, comprehension, and transcription.",
    },
    level: { fr: "Intermédiaire · B1 interne", en: "Intermediate · internal B1" },
    modules: [
      { fr: "Compréhension orale — structure", en: "Listening comprehension — structure" },
      { fr: "Dictée — structure", en: "Dictation — structure" },
      { fr: "Révision — structure", en: "Review — structure" },
    ],
  },
] as const satisfies readonly {
  id: string;
  title: LocalizedText;
  description: LocalizedText;
  level: LocalizedText;
  modules: readonly LocalizedText[];
}[];

export type DemoCourse = (typeof DEMO_COURSES)[number];

export function getDemoCourse(id: string): DemoCourse | undefined {
  return DEMO_COURSES.find((course) => course.id === id);
}

export const DEMO_LESSONS = [
  {
    id: "dire-bonjour",
    title: { fr: "Dire bonjour", en: "Saying hello" },
    objective: {
      fr: "Comprendre la structure d’une future leçon de salutation.",
      en: "Understand the structure of a future greeting lesson.",
    },
  },
  {
    id: "demander-comment-ca-va",
    title: { fr: "Demander comment quelqu’un va", en: "Asking how someone is" },
    objective: {
      fr: "Prévisualiser un parcours de dialogue sans contenu linguistique inventé.",
      en: "Preview a dialogue journey without invented language content.",
    },
  },
] as const satisfies readonly {
  id: string;
  title: LocalizedText;
  objective: LocalizedText;
}[];

export function getDemoLesson(id: string) {
  return DEMO_LESSONS.find((lesson) => lesson.id === id);
}

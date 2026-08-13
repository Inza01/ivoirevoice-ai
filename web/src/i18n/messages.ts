import type { CapabilityStatus, UiLocale } from "@/lib/languages/registry";

export interface MessageCatalog {
  readonly brand: {
    readonly name: string;
    readonly tagline: string;
  };
  readonly navigation: {
    readonly skipToContent: string;
    readonly home: string;
    readonly transcribe: string;
    readonly translate: string;
    readonly learn: string;
    readonly practice: string;
    readonly community: string;
    readonly profile: string;
    readonly about: string;
    readonly openMenu: string;
    readonly closeMenu: string;
    readonly primaryNavigation: string;
    readonly interfaceLanguage: string;
  };
  readonly status: Readonly<Record<CapabilityStatus, string>>;
  readonly common: {
    readonly backHome: string;
    readonly learnMore: string;
    readonly retry: string;
    readonly clear: string;
    readonly copy: string;
    readonly loading: string;
    readonly unavailable: string;
    readonly demoNeedsReview: string;
    readonly or: string;
    readonly previous: string;
    readonly next: string;
    readonly duration: string;
    readonly processingTime: string;
    readonly detectedLanguage: string;
    readonly notFoundTitle: string;
    readonly notFoundDescription: string;
    readonly autoLanguage: string;
    readonly structureOnly: string;
    readonly lessonStructureDescription: string;
    readonly progressLocal: string;
  };
  readonly errors: {
    readonly generic: string;
    readonly network: string;
    readonly unavailable: string;
    readonly invalidFile: string;
    readonly fileTooLarge: string;
    readonly incompatibleModelLanguage: string;
    readonly modelUnavailable: string;
    readonly transcriptionFailed: string;
    readonly unknownLanguage: string;
    readonly unknownModel: string;
    readonly copyFailed: string;
    readonly microphoneDenied: string;
  };
  readonly footer: {
    readonly description: string;
    readonly experimental: string;
    readonly explore: string;
    readonly secondaryNavigation: string;
    readonly contact: string;
    readonly contactPending: string;
    readonly localResearch: string;
    readonly madeInCoteDIvoire: string;
  };
  readonly uploader: {
    readonly title: string;
    readonly description: string;
    readonly dropActive: string;
    readonly formats: string;
    readonly chooseFile: string;
    readonly selectedFile: string;
    readonly microphoneTitle: string;
    readonly microphoneDescription: string;
    readonly record: string;
    readonly microphoneUnavailable: string;
    readonly privacyNotice: string;
  };
  readonly forms: {
    readonly sourceLanguage: string;
    readonly targetLanguage: string;
    readonly audioLanguage: string;
    readonly audioModel: string;
    readonly interfaceLanguage: string;
    readonly chooseLanguage: string;
    readonly chooseModel: string;
    readonly textToTranslate: string;
    readonly translationResult: string;
    readonly required: string;
    readonly optional: string;
    readonly submit: string;
  };
  readonly home: {
    readonly eyebrow: string;
    readonly title: string;
    readonly description: string;
    readonly primaryAction: string;
    readonly secondaryAction: string;
    readonly localPrivacyNote: string;
    readonly visualCaption: string;
    readonly featureTitle: string;
    readonly featureDescription: string;
    readonly transcribeTitle: string;
    readonly transcribeDescription: string;
    readonly translateTitle: string;
    readonly translateDescription: string;
    readonly learnTitle: string;
    readonly learnDescription: string;
    readonly languagesTitle: string;
    readonly languagesDescription: string;
    readonly futureLanguages: string;
    readonly coursesTitle: string;
    readonly coursesDescription: string;
    readonly technologyTitle: string;
    readonly technologyDescription: string;
  };
  readonly transcribe: {
    readonly eyebrow: string;
    readonly title: string;
    readonly description: string;
    readonly integrationNotice: string;
    readonly discoveryLoading: string;
    readonly discoveryError: string;
    readonly action: string;
    readonly resultTitle: string;
    readonly resultEmptyTitle: string;
    readonly resultEmptyDescription: string;
    readonly readyTitle: string;
    readonly readyDescription: string;
    readonly processingTitle: string;
    readonly processingDescription: string;
    readonly successTitle: string;
    readonly copySuccess: string;
    readonly translateAction: string;
    readonly downloadTxt: string;
    readonly downloadJson: string;
    readonly deleteNotice: string;
  };
  readonly translate: {
    readonly eyebrow: string;
    readonly title: string;
    readonly description: string;
    readonly comingSoonNotice: string;
    readonly sourcePlaceholder: string;
    readonly targetPlaceholder: string;
    readonly useTranscription: string;
    readonly swapLanguages: string;
    readonly action: string;
    readonly listen: string;
    readonly providerNotice: string;
  };
  readonly learn: {
    readonly eyebrow: string;
    readonly title: string;
    readonly description: string;
    readonly welcomeGuest: string;
    readonly continueTitle: string;
    readonly progressTitle: string;
    readonly currentCoursesTitle: string;
    readonly recommendedCoursesTitle: string;
    readonly levelTitle: string;
    readonly levelDisclaimer: string;
    readonly signInNotice: string;
    readonly browseCourses: string;
    readonly previewObjective: string;
  };
  readonly courses: {
    readonly eyebrow: string;
    readonly title: string;
    readonly description: string;
    readonly filterLevel: string;
    readonly filterTopic: string;
    readonly filterLanguage: string;
    readonly allFilters: string;
    readonly filtersLabel: string;
    readonly demoStatus: string;
    readonly openCourse: string;
    readonly emptyTitle: string;
    readonly emptyDescription: string;
  };
  readonly courseDetail: {
    readonly eyebrow: string;
    readonly objectives: string;
    readonly modules: string;
    readonly lessons: string;
    readonly start: string;
    readonly continue: string;
    readonly progressUnavailable: string;
    readonly notFoundTitle: string;
    readonly notFoundDescription: string;
  };
  readonly lesson: {
    readonly eyebrow: string;
    readonly objective: string;
    readonly content: string;
    readonly dioulaExample: string;
    readonly frenchTranslation: string;
    readonly englishTranslation: string;
    readonly listen: string;
    readonly vocabulary: string;
    readonly quickExercise: string;
    readonly markComplete: string;
    readonly completionUnavailable: string;
    readonly notFoundTitle: string;
    readonly notFoundDescription: string;
  };
  readonly practice: {
    readonly eyebrow: string;
    readonly title: string;
    readonly description: string;
    readonly exerciseTypes: string;
    readonly multipleChoice: string;
    readonly fillBlank: string;
    readonly matching: string;
    readonly listeningChoice: string;
    readonly listeningTranscription: string;
    readonly frenchToDioula: string;
    readonly dioulaToFrench: string;
    readonly pronunciation: string;
    readonly pronunciationDisclaimer: string;
    readonly validate: string;
    readonly correct: string;
    readonly review: string;
    readonly expectedAnswer: string;
    readonly explanation: string;
  };
  readonly community: {
    readonly eyebrow: string;
    readonly title: string;
    readonly description: string;
    readonly comingSoonNotice: string;
    readonly moderationNotice: string;
    readonly categoriesTitle: string;
    readonly learning: string;
    readonly pronunciation: string;
    readonly vocabulary: string;
    readonly culture: string;
    readonly translation: string;
  };
  readonly profile: {
    readonly eyebrow: string;
    readonly title: string;
    readonly description: string;
    readonly guestTitle: string;
    readonly guestDescription: string;
    readonly signIn: string;
    readonly preferences: string;
    readonly studiedLanguage: string;
    readonly level: string;
    readonly progress: string;
    readonly courses: string;
    readonly recentActivity: string;
    readonly privacy: string;
    readonly deleteData: string;
    readonly authenticationUnavailable: string;
    readonly plannedValue: string;
  };
  readonly about: {
    readonly eyebrow: string;
    readonly title: string;
    readonly description: string;
    readonly missionTitle: string;
    readonly missionDescription: string;
    readonly capabilitiesTitle: string;
    readonly resultsTitle: string;
    readonly resultsDescription: string;
    readonly privacyTitle: string;
    readonly privacyDescription: string;
    readonly limitationsTitle: string;
    readonly limitationsDescription: string;
    readonly futureTitle: string;
    readonly futureDescription: string;
    readonly frozenModelLabel: string;
    readonly holdoutLabel: string;
    readonly werLabel: string;
    readonly cerLabel: string;
    readonly rtfLabel: string;
  };
}

export const MESSAGES: Readonly<Record<UiLocale, MessageCatalog>> = {
  fr: {
    brand: {
      name: "IvoireVoice",
      tagline: "Écoutez. Traduisez. Apprenez les langues africaines.",
    },
    navigation: {
      skipToContent: "Aller au contenu principal",
      home: "Accueil",
      transcribe: "Transcrire",
      translate: "Traduire",
      learn: "Apprendre",
      practice: "S'exercer",
      community: "Communauté",
      profile: "Mon espace",
      about: "À propos",
      openMenu: "Ouvrir le menu",
      closeMenu: "Fermer le menu",
      primaryNavigation: "Navigation principale",
      interfaceLanguage: "Langue de l'interface",
    },
    status: {
      available: "Disponible",
      experimental: "Expérimental",
      coming_soon: "Bientôt disponible",
    },
    common: {
      backHome: "Retour à l'accueil",
      learnMore: "En savoir plus",
      retry: "Réessayer",
      clear: "Effacer",
      copy: "Copier",
      loading: "Chargement en cours",
      unavailable: "Cette fonctionnalité n'est pas encore disponible.",
      demoNeedsReview: "DÉMO — À VALIDER PAR UN EXPERT LINGUISTIQUE",
      or: "ou",
      previous: "Précédent",
      next: "Suivant",
      duration: "Durée",
      processingTime: "Temps de traitement",
      detectedLanguage: "Langue détectée",
      notFoundTitle: "Page introuvable",
      notFoundDescription: "Cette adresse ne correspond à aucune page de la fondation IvoireVoice.",
      autoLanguage: "Auto",
      structureOnly: "Structure de démonstration, sans contenu linguistique publié.",
      lessonStructureDescription:
        "Cette page montre la hiérarchie de lecture et les espaces réservés à l'audio, au vocabulaire et à l'exercice. Aucun exemple dioula n'est publié avant validation linguistique.",
      progressLocal: "Progression locale",
    },
    errors: {
      generic: "Une erreur est survenue. Vous pouvez réessayer.",
      network: "Le service est injoignable. Vérifiez votre connexion puis réessayez.",
      unavailable: "Le service est temporairement indisponible.",
      invalidFile: "Ce fichier n'est pas un audio valide.",
      fileTooLarge: "Le fichier dépasse la taille autorisée.",
      incompatibleModelLanguage: "La langue choisie n'est pas compatible avec ce modèle.",
      modelUnavailable: "Le modèle sélectionné est momentanément indisponible.",
      transcriptionFailed: "La transcription est impossible pour le moment.",
      unknownLanguage: "La langue choisie n'est pas reconnue par le service.",
      unknownModel: "Le modèle choisi n'est pas reconnu par le service.",
      copyFailed: "Impossible de copier le texte. Sélectionnez-le manuellement.",
      microphoneDenied: "L'accès au microphone a été refusé.",
    },
    footer: {
      description: "Une plateforme pour écouter, transcrire et apprendre.",
      experimental: "Les capacités linguistiques restent expérimentales ou à venir.",
      explore: "Explorer",
      secondaryNavigation: "Navigation secondaire",
      contact: "Contact",
      contactPending: "Contact à configurer",
      localResearch: "IvoireVoice — recherche locale.",
      madeInCoteDIvoire: "Conçu en Côte d'Ivoire pour valoriser les langues africaines.",
    },
    uploader: {
      title: "Importer un audio",
      description: "Glissez-déposez un fichier audio ici ou choisissez-le sur votre appareil.",
      dropActive: "Déposez le fichier audio.",
      formats: "WAV, FLAC, OGG ou MP3 — 25 Mo et 30 secondes maximum",
      chooseFile: "Choisir un fichier",
      selectedFile: "Fichier sélectionné",
      microphoneTitle: "Enregistrer avec le microphone",
      microphoneDescription: "Créez un nouvel enregistrement depuis cet appareil.",
      record: "Commencer l'enregistrement",
      microphoneUnavailable: "L'enregistrement sera étudié dans une phase dédiée.",
      privacyNotice:
        "L'audio est traité temporairement, sans stockage permanent ni utilisation pour l'entraînement.",
    },
    forms: {
      sourceLanguage: "Langue source",
      targetLanguage: "Langue cible",
      audioLanguage: "Langue de l'audio",
      audioModel: "Modèle de transcription",
      interfaceLanguage: "Langue de l'interface",
      chooseLanguage: "Choisir une langue",
      chooseModel: "Choisir un modèle",
      textToTranslate: "Texte à traduire",
      translationResult: "Résultat traduit",
      required: "Obligatoire",
      optional: "Facultatif",
      submit: "Valider",
    },
    home: {
      eyebrow: "Plateforme linguistique africaine",
      title: "Écoutez. Traduisez. Apprenez les langues africaines.",
      description:
        "IvoireVoice prépare une expérience simple pour la transcription et l'apprentissage.",
      primaryAction: "Transcrire un audio",
      secondaryAction: "Apprendre le Dioula",
      localPrivacyNote: "Vos audios restent sous votre contrôle.",
      visualCaption: "La voix ouvre le chemin vers la langue.",
      featureTitle: "Des outils linguistiques réunis au même endroit",
      featureDescription: "Chaque capacité affiche clairement son niveau de disponibilité.",
      transcribeTitle: "Transcrire",
      transcribeDescription: "Transformer un audio autorisé en texte dans la même langue.",
      translateTitle: "Traduire",
      translateDescription: "Préparer un passage explicite d'une langue à une autre.",
      learnTitle: "Apprendre",
      learnDescription: "Découvrir des cours, leçons et exercices pédagogiques.",
      languagesTitle: "Trois langues au cœur de la vision",
      languagesDescription: "Français, English et Dioula, avec des statuts transparents.",
      futureLanguages: "D'autres langues africaines arriveront progressivement.",
      coursesTitle: "Aperçu des cours",
      coursesDescription: "Les contenus affichés sont des démonstrations à faire valider.",
      technologyTitle: "Une technologie au service de la langue",
      technologyDescription:
        "IA vocale, transcription et apprentissage interactif, sans jargon inutile.",
    },
    transcribe: {
      eyebrow: "Audio vers texte",
      title: "Transcrire un audio",
      description: "Importez un fichier audio autorisé pour obtenir une transcription.",
      integrationNotice:
        "La transcription locale est expérimentale. Les résultats peuvent contenir des erreurs.",
      discoveryLoading: "Chargement des langues et modèles disponibles.",
      discoveryError: "Impossible de charger les capacités du service de transcription.",
      action: "Transcrire",
      resultTitle: "Transcription",
      resultEmptyTitle: "Votre transcription apparaîtra ici",
      resultEmptyDescription: "Aucun audio n'a encore été envoyé au service.",
      readyTitle: "Audio prêt",
      readyDescription: "Choisissez une langue et un modèle, puis lancez la transcription.",
      processingTitle: "Transcription en cours",
      processingDescription: "Le fichier est traité temporairement. Ne fermez pas cette page.",
      successTitle: "Transcription terminée",
      copySuccess: "Texte copié.",
      translateAction: "Traduire cette transcription",
      downloadTxt: "Télécharger TXT",
      downloadJson: "Télécharger JSON",
      deleteNotice: "L'audio temporaire est supprimé automatiquement après le traitement.",
    },
    translate: {
      eyebrow: "Texte vers texte",
      title: "Traduire un texte",
      description: "Choisissez explicitement une langue source et une langue cible.",
      comingSoonNotice: "Aucun moteur de traduction n'est encore intégré ou validé.",
      sourcePlaceholder: "Saisissez le texte source.",
      targetPlaceholder: "La traduction apparaîtra ici lorsqu'un service sera disponible.",
      useTranscription: "Utiliser une transcription",
      swapLanguages: "Inverser les langues",
      action: "Traduire",
      listen: "Écouter",
      providerNotice: "Le fournisseur et le statut seront indiqués avec chaque résultat.",
    },
    learn: {
      eyebrow: "Apprentissage",
      title: "Apprendre le Dioula pas à pas",
      description: "Explorez la future structure des cours, modules et leçons.",
      welcomeGuest: "Bienvenue dans votre espace d'apprentissage",
      continueTitle: "Continuez votre apprentissage",
      progressTitle: "Progression globale",
      currentCoursesTitle: "Cours en cours",
      recommendedCoursesTitle: "Cours recommandés",
      levelTitle: "Niveau actuel",
      levelDisclaimer: "Les niveaux sont internes et ne constituent pas une certification CECR.",
      signInNotice: "La progression persistante nécessitera un compte.",
      browseCourses: "Voir les cours",
      previewObjective:
        "Prévisualiser un parcours progressif et ses états éditoriaux avant validation experte.",
    },
    courses: {
      eyebrow: "Catalogue",
      title: "Choisir un cours",
      description: "Parcourez des structures de cours de démonstration.",
      filterLevel: "Filtrer par niveau",
      filterTopic: "Filtrer par thème",
      filterLanguage: "Filtrer par langue",
      allFilters: "Tous",
      filtersLabel: "Filtres du catalogue",
      demoStatus: "Contenu de démonstration à valider",
      openCourse: "Voir le cours",
      emptyTitle: "Aucun cours ne correspond à ces filtres",
      emptyDescription: "Modifiez un filtre pour voir d'autres cours de démonstration.",
    },
    courseDetail: {
      eyebrow: "Cours",
      objectives: "Objectifs",
      modules: "Modules",
      lessons: "Leçons",
      start: "Commencer le cours",
      continue: "Continuer le cours",
      progressUnavailable: "La sauvegarde de progression n'est pas encore disponible.",
      notFoundTitle: "Cours introuvable",
      notFoundDescription: "Ce cours de démonstration n'existe pas ou n'est plus disponible.",
    },
    lesson: {
      eyebrow: "Leçon",
      objective: "Objectif pédagogique",
      content: "Contenu",
      dioulaExample: "Exemple en dioula",
      frenchTranslation: "Traduction française",
      englishTranslation: "Traduction anglaise",
      listen: "Écouter l'exemple",
      vocabulary: "Vocabulaire important",
      quickExercise: "Exercice rapide",
      markComplete: "Marquer comme terminée",
      completionUnavailable: "Cette action nécessitera un compte et une progression active.",
      notFoundTitle: "Leçon introuvable",
      notFoundDescription: "Cette leçon de démonstration n'existe pas ou n'est plus disponible.",
    },
    practice: {
      eyebrow: "Entraînement",
      title: "S'exercer à son rythme",
      description: "Découvrez les formats d'exercices prévus pour la plateforme.",
      exerciseTypes: "Types d'exercices",
      multipleChoice: "Question à choix multiple",
      fillBlank: "Texte à compléter",
      matching: "Associer un mot et sa traduction",
      listeningChoice: "Écouter puis choisir",
      listeningTranscription: "Écouter puis transcrire",
      frenchToDioula: "Français vers dioula",
      dioulaToFrench: "Dioula vers français",
      pronunciation: "Pratique de la prononciation",
      pronunciationDisclaimer: "Aucun score phonétique scientifique n'est disponible.",
      validate: "Vérifier ma réponse",
      correct: "Correct",
      review: "À revoir",
      expectedAnswer: "Bonne réponse",
      explanation: "Explication",
    },
    community: {
      eyebrow: "Échanges",
      title: "Communauté IvoireVoice",
      description: "Un futur espace pour poser des questions et partager des usages.",
      comingSoonNotice: "La communauté ouvrira après la mise en place de la modération.",
      moderationNotice: "Aucune discussion ni identité fictive n'est affichée aujourd'hui.",
      categoriesTitle: "Catégories prévues",
      learning: "Apprentissage",
      pronunciation: "Prononciation",
      vocabulary: "Vocabulaire",
      culture: "Culture",
      translation: "Traduction",
    },
    profile: {
      eyebrow: "Mon espace",
      title: "Profil et progression",
      description: "Retrouvez à terme vos préférences et votre parcours.",
      guestTitle: "Votre espace personnel n'est pas encore actif",
      guestDescription: "L'authentification sera introduite dans une phase dédiée.",
      signIn: "Se connecter",
      preferences: "Préférences",
      studiedLanguage: "Langue étudiée",
      level: "Niveau interne",
      progress: "Progression",
      courses: "Cours suivis",
      recentActivity: "Activité récente",
      privacy: "Confidentialité",
      deleteData: "Supprimer mes données",
      authenticationUnavailable: "Aucun faux compte ou historique n'est créé.",
      plannedValue: "prévu",
    },
    about: {
      eyebrow: "À propos",
      title: "IvoireVoice, une plateforme linguistique en construction",
      description: "Le produit réunit une recherche ASR locale et une vision éducative.",
      missionTitle: "Notre mission",
      missionDescription: "Rendre les langues africaines plus accessibles par des outils utiles.",
      capabilitiesTitle: "Capacités et statuts",
      resultsTitle: "Résultats ASR publics",
      resultsDescription:
        "Les résultats affichés sont des agrégats gelés, jamais une nouvelle évaluation.",
      privacyTitle: "Confidentialité",
      privacyDescription: "Le corpus, les transcriptions et le checkpoint restent hors du web.",
      limitationsTitle: "Limites",
      limitationsDescription:
        "Le modèle dioula reste expérimental et spécialisé au contexte local.",
      futureTitle: "Perspectives",
      futureDescription: "Étendre progressivement le produit à d'autres langues africaines.",
      frozenModelLabel: "Modèle Dioula final gelé",
      holdoutLabel: "audios · 3 locuteurs · holdout scellé",
      werLabel: "WER micro",
      cerLabel: "CER micro",
      rtfLabel: "RTF",
    },
  },
  en: {
    brand: {
      name: "IvoireVoice",
      tagline: "Listen. Translate. Learn African languages.",
    },
    navigation: {
      skipToContent: "Skip to main content",
      home: "Home",
      transcribe: "Transcribe",
      translate: "Translate",
      learn: "Learn",
      practice: "Practice",
      community: "Community",
      profile: "My space",
      about: "About",
      openMenu: "Open menu",
      closeMenu: "Close menu",
      primaryNavigation: "Primary navigation",
      interfaceLanguage: "Interface language",
    },
    status: {
      available: "Available",
      experimental: "Experimental",
      coming_soon: "Coming soon",
    },
    common: {
      backHome: "Back to home",
      learnMore: "Learn more",
      retry: "Try again",
      clear: "Clear",
      copy: "Copy",
      loading: "Loading",
      unavailable: "This feature is not available yet.",
      demoNeedsReview: "DEMO — TO BE REVIEWED BY A LANGUAGE EXPERT",
      or: "or",
      previous: "Previous",
      next: "Next",
      duration: "Duration",
      processingTime: "Processing time",
      detectedLanguage: "Detected language",
      notFoundTitle: "Page not found",
      notFoundDescription: "This address does not match a page in the IvoireVoice Foundation.",
      autoLanguage: "Auto",
      structureOnly: "Demonstration structure with no published language content.",
      lessonStructureDescription:
        "This page shows the reading hierarchy and the spaces reserved for audio, vocabulary, and an exercise. No Dioula example is published before language review.",
      progressLocal: "Local progress",
    },
    errors: {
      generic: "Something went wrong. You can try again.",
      network: "The service cannot be reached. Check your connection and try again.",
      unavailable: "The service is temporarily unavailable.",
      invalidFile: "This file is not valid audio.",
      fileTooLarge: "The file exceeds the allowed size.",
      incompatibleModelLanguage: "The selected language is not compatible with this model.",
      modelUnavailable: "The selected model is temporarily unavailable.",
      transcriptionFailed: "Transcription is not possible right now.",
      unknownLanguage: "The selected language is not recognized by the service.",
      unknownModel: "The selected model is not recognized by the service.",
      copyFailed: "The text could not be copied. Select it manually instead.",
      microphoneDenied: "Microphone access was denied.",
    },
    footer: {
      description: "A platform for listening, transcribing, and learning.",
      experimental: "Language capabilities remain experimental or are coming soon.",
      explore: "Explore",
      secondaryNavigation: "Secondary navigation",
      contact: "Contact",
      contactPending: "Contact to be configured",
      localResearch: "IvoireVoice — local research.",
      madeInCoteDIvoire: "Designed in Côte d'Ivoire to support African languages.",
    },
    uploader: {
      title: "Upload audio",
      description: "Drag and drop an audio file here or choose one from your device.",
      dropActive: "Drop the audio file.",
      formats: "WAV, FLAC, OGG, or MP3 — maximum 25 MB and 30 seconds",
      chooseFile: "Choose a file",
      selectedFile: "Selected file",
      microphoneTitle: "Record with the microphone",
      microphoneDescription: "Create a new recording from this device.",
      record: "Start recording",
      microphoneUnavailable: "Recording will be considered in a dedicated phase.",
      privacyNotice:
        "Uploaded audio is processed temporarily, with no permanent storage or use for training.",
    },
    forms: {
      sourceLanguage: "Source language",
      targetLanguage: "Target language",
      audioLanguage: "Audio language",
      audioModel: "Transcription model",
      interfaceLanguage: "Interface language",
      chooseLanguage: "Choose a language",
      chooseModel: "Choose a model",
      textToTranslate: "Text to translate",
      translationResult: "Translated result",
      required: "Required",
      optional: "Optional",
      submit: "Submit",
    },
    home: {
      eyebrow: "African language platform",
      title: "Listen. Translate. Learn African languages.",
      description: "IvoireVoice is preparing a simple transcription and learning experience.",
      primaryAction: "Transcribe audio",
      secondaryAction: "Learn Dioula",
      localPrivacyNote: "Your audio stays under your control.",
      visualCaption: "Voice opens a path to language.",
      featureTitle: "Language tools brought together in one place",
      featureDescription: "Every capability clearly states its availability.",
      transcribeTitle: "Transcribe",
      transcribeDescription: "Turn authorized audio into text in the same language.",
      translateTitle: "Translate",
      translateDescription: "Prepare an explicit passage from one language to another.",
      learnTitle: "Learn",
      learnDescription: "Discover educational courses, lessons, and exercises.",
      languagesTitle: "Three languages at the heart of the vision",
      languagesDescription: "French, English, and Dioula, with transparent status labels.",
      futureLanguages: "Other African languages will be added progressively.",
      coursesTitle: "Course preview",
      coursesDescription: "Displayed content is demonstration material requiring review.",
      technologyTitle: "Technology supporting language",
      technologyDescription:
        "Speech AI, transcription, and interactive learning without unnecessary jargon.",
    },
    transcribe: {
      eyebrow: "Audio to text",
      title: "Transcribe audio",
      description: "Upload an authorized audio file to obtain a transcription.",
      integrationNotice: "Local transcription is experimental. Results may contain errors.",
      discoveryLoading: "Loading available languages and models.",
      discoveryError: "The transcription service capabilities could not be loaded.",
      action: "Transcribe",
      resultTitle: "Transcription",
      resultEmptyTitle: "Your transcription will appear here",
      resultEmptyDescription: "No audio has been sent to the service yet.",
      readyTitle: "Audio ready",
      readyDescription: "Choose a language and model, then start transcription.",
      processingTitle: "Transcription in progress",
      processingDescription: "The file is being processed temporarily. Keep this page open.",
      successTitle: "Transcription complete",
      copySuccess: "Text copied.",
      translateAction: "Translate this transcription",
      downloadTxt: "Download TXT",
      downloadJson: "Download JSON",
      deleteNotice: "Temporary audio is automatically deleted after processing.",
    },
    translate: {
      eyebrow: "Text to text",
      title: "Translate text",
      description: "Explicitly choose a source language and a target language.",
      comingSoonNotice: "No translation engine has been integrated or validated yet.",
      sourcePlaceholder: "Enter the source text.",
      targetPlaceholder: "The translation will appear here when a service is available.",
      useTranscription: "Use a transcription",
      swapLanguages: "Swap languages",
      action: "Translate",
      listen: "Listen",
      providerNotice: "The provider and status will be shown with every result.",
    },
    learn: {
      eyebrow: "Learning",
      title: "Learn Dioula step by step",
      description: "Explore the planned course, module, and lesson structure.",
      welcomeGuest: "Welcome to your learning space",
      continueTitle: "Continue learning",
      progressTitle: "Overall progress",
      currentCoursesTitle: "Current courses",
      recommendedCoursesTitle: "Recommended courses",
      levelTitle: "Current level",
      levelDisclaimer: "Levels are internal and do not represent CEFR certification.",
      signInNotice: "Persistent progress will require an account.",
      browseCourses: "Browse courses",
      previewObjective:
        "Preview a progressive journey and its editorial states before expert review.",
    },
    courses: {
      eyebrow: "Catalog",
      title: "Choose a course",
      description: "Browse demonstration course structures.",
      filterLevel: "Filter by level",
      filterTopic: "Filter by topic",
      filterLanguage: "Filter by language",
      allFilters: "All",
      filtersLabel: "Course catalog filters",
      demoStatus: "Demonstration content requiring review",
      openCourse: "View course",
      emptyTitle: "No course matches these filters",
      emptyDescription: "Change a filter to view other demonstration courses.",
    },
    courseDetail: {
      eyebrow: "Course",
      objectives: "Objectives",
      modules: "Modules",
      lessons: "Lessons",
      start: "Start course",
      continue: "Continue course",
      progressUnavailable: "Progress storage is not available yet.",
      notFoundTitle: "Course not found",
      notFoundDescription: "This demonstration course does not exist or is unavailable.",
    },
    lesson: {
      eyebrow: "Lesson",
      objective: "Learning objective",
      content: "Content",
      dioulaExample: "Dioula example",
      frenchTranslation: "French translation",
      englishTranslation: "English translation",
      listen: "Listen to the example",
      vocabulary: "Key vocabulary",
      quickExercise: "Quick exercise",
      markComplete: "Mark as complete",
      completionUnavailable: "This action will require an account and active progress.",
      notFoundTitle: "Lesson not found",
      notFoundDescription: "This demonstration lesson does not exist or is unavailable.",
    },
    practice: {
      eyebrow: "Practice",
      title: "Practice at your own pace",
      description: "Discover the exercise formats planned for the platform.",
      exerciseTypes: "Exercise types",
      multipleChoice: "Multiple-choice question",
      fillBlank: "Fill in the blank",
      matching: "Match a word and its translation",
      listeningChoice: "Listen and choose",
      listeningTranscription: "Listen and transcribe",
      frenchToDioula: "French to Dioula",
      dioulaToFrench: "Dioula to French",
      pronunciation: "Pronunciation practice",
      pronunciationDisclaimer: "No scientifically validated pronunciation score is available.",
      validate: "Check my answer",
      correct: "Correct",
      review: "Review needed",
      expectedAnswer: "Correct answer",
      explanation: "Explanation",
    },
    community: {
      eyebrow: "Conversations",
      title: "IvoireVoice Community",
      description: "A future space for questions and shared language usage.",
      comingSoonNotice: "Community features will open after moderation is in place.",
      moderationNotice: "No fictional discussions or identities are displayed today.",
      categoriesTitle: "Planned categories",
      learning: "Learning",
      pronunciation: "Pronunciation",
      vocabulary: "Vocabulary",
      culture: "Culture",
      translation: "Translation",
    },
    profile: {
      eyebrow: "My space",
      title: "Profile and progress",
      description: "Your preferences and learning journey will eventually appear here.",
      guestTitle: "Your personal space is not active yet",
      guestDescription: "Authentication will be introduced in a dedicated phase.",
      signIn: "Sign in",
      preferences: "Preferences",
      studiedLanguage: "Studied language",
      level: "Internal level",
      progress: "Progress",
      courses: "Enrolled courses",
      recentActivity: "Recent activity",
      privacy: "Privacy",
      deleteData: "Delete my data",
      authenticationUnavailable: "No fake account or history is created.",
      plannedValue: "planned",
    },
    about: {
      eyebrow: "About",
      title: "IvoireVoice, a language platform in development",
      description: "The product combines local ASR research with an educational vision.",
      missionTitle: "Our mission",
      missionDescription: "Make African languages more accessible through useful tools.",
      capabilitiesTitle: "Capabilities and status",
      resultsTitle: "Public ASR results",
      resultsDescription: "Displayed results are frozen aggregates, never a new evaluation.",
      privacyTitle: "Privacy",
      privacyDescription: "The corpus, transcriptions, and checkpoint remain outside the web.",
      limitationsTitle: "Limitations",
      limitationsDescription:
        "The Dioula model remains experimental and specialized to its local context.",
      futureTitle: "Future directions",
      futureDescription: "Progressively extend the product to other African languages.",
      frozenModelLabel: "Frozen final Dioula model",
      holdoutLabel: "audio files · 3 speakers · sealed holdout",
      werLabel: "Micro WER",
      cerLabel: "Micro CER",
      rtfLabel: "RTF",
    },
  },
};

export function getMessages(locale: UiLocale): MessageCatalog {
  return MESSAGES[locale];
}

export const enum RetrievalMode {
    Hybrid = "hybrid",
    Vectors = "vectors",
    Text = "text"
}

export type AskRequestOverrides = {
    retrievalMode?: RetrievalMode;
    semanticRanker?: boolean;
    semanticCaptions?: boolean;
    excludeCategory?: string;
    top?: number;
    temperature?: number;
    promptTemplate?: string;
    promptTemplatePrefix?: string;
    promptTemplateSuffix?: string;
    suggestFollowupQuestions?: boolean;
//=======
    expectcodeoutput?: boolean;
    useOidSecurityFilter?: boolean;
    useGroupsSecurityFilter?: boolean;
};

export type AskRequest = {
    question: string;
    overrides?: AskRequestOverrides;
    idToken?: string;
};

export type AskResponse = {
    answer: string;
    thoughts: string | null;
    data_points: string[];
    error?: string;
};

export type ChatTurn = {
    user: string;
    bot?: string;
};

export type ChatRequest = {
    history: ChatTurn[];
    overrides?: AskRequestOverrides;
    idToken?: string;
    shouldStream?: boolean;
};

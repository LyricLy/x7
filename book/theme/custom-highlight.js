const REPR = [
    {
        scope: "number",
        match: /-?\b\d+(\.(\d*\(\d+\)|\d+)|([+\-]\d+)?\/\d+)?/,
    },
];

hljs.registerLanguage("x7", (hljs) => ({
    name: "x7",
    keywords: {
        $pattern: /./,
        keyword: "W F M T C _ l ~ e s ! m w Z ` { }",
    },
    contains: [
        {
            scope: "number",
            match: /\d+/,
        },
        {
            scope: "variable",
            match: /[;:][^\d]/,
        },
        {
            scope: "title.function.invoke",
            match: /;[1-9]\d*/,
        },
        {
            beginScope: "comment",
            end: /$/,
            contains: REPR,
            variants: [
                {
                    begin: /^>(?=[^\n<]*$)/,
                    contains: REPR,
                },
                {
                    scope: "comment",
                    begin: /^>/,
                    contains: [
                        {
                            scope: "meta",
                            begin: "<",
                            end: ">",
                            contains: REPR,
                        },
                    ],
                },
            ],
        },
    ],
}));

// reinit hljs to force it to acknowledge our language
hljs.initHighlightingOnLoad();

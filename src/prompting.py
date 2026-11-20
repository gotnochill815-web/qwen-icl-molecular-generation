"""Shared molecular-generation prompt formatting."""

def format_chat_prompt(tokenizer, prompt, properties=None, targets=None):
    def get_property_names(value):
        if isinstance(value, dict):
            items = value.keys()
        elif isinstance(value, str):
            items = [value]
        elif isinstance(value, (list, tuple, set)):
            items = value
        else:
            items = []

        return {
            str(item).strip().lower()
            for item in items
            if str(item).strip()
        }

    names = (
        get_property_names(properties)
        | get_property_names(targets)
    )

    if names == {"logp"}:
        system_prompt = (
            "You are a molecular generation model. "
            "Generate exactly ONE valid molecular SMILES string whose "
            "calculated octanol/water partition coefficient (LogP), "
            "calculated using RDKit Crippen MolLogP, is as close as "
            "possible to the requested target. A valid SMILES is not "
            "sufficient by itself: try to match the requested LogP while "
            "producing a chemically plausible molecule. Treat the examples "
            "as demonstrations of the requested input-output format. "
            "For each example, the supplied LogP is the reference property "
            "value for that example. Do not copy an example unnecessarily. "
            "Return only the generated SMILES. Do not include explanations, "
            "Markdown fences, labels, or multiple candidates."
        )
    elif names == {"qed"}:
        system_prompt = (
            "You are a molecular generation model. "
            "Generate exactly ONE valid molecular SMILES string whose "
            "quantitative estimate of drug-likeness (QED), calculated "
            "using RDKit QED.qed(), is as close as possible to the "
            "requested target. A valid SMILES is not sufficient by itself: "
            "try to match the requested QED while producing a chemically "
            "plausible molecule. Treat the examples as demonstrations of "
            "the requested input-output format. For each example, the "
            "supplied QED is the reference property value for that example. "
            "Do not copy an example unnecessarily. Return only the "
            "generated SMILES. Do not include explanations, Markdown "
            "fences, labels, or multiple candidates."
        )
    elif names == {"tpsa"}:
        system_prompt = (
            "You are a molecular generation model. "
            "Generate exactly ONE valid molecular SMILES string whose "
            "topological polar surface area (TPSA), calculated using "
            "RDKit rdMolDescriptors.CalcTPSA, is as close as possible "
            "to the requested target. A valid SMILES is not sufficient "
            "by itself: try to match the requested TPSA while producing "
            "a chemically plausible molecule. Treat the examples as "
            "demonstrations of the requested input-output format. "
            "For each example, the supplied TPSA is the reference property "
            "value for that example. Do not copy an example unnecessarily. "
            "Return only the generated SMILES. Do not include explanations, "
            "Markdown fences, labels, or multiple candidates."
        )
    elif names == {"sas"}:
        system_prompt = (
            "You are a molecular generation model. "
            "Generate exactly ONE valid molecular SMILES string whose "
            "synthetic accessibility score (SAS), calculated using "
            "RDKit sascorer.calculateScore, is as close as possible "
            "to the requested target. A valid SMILES is not sufficient "
            "by itself: try to match the requested SAS while producing "
            "a chemically plausible molecule. Treat the examples as "
            "demonstrations of the requested input-output format. "
            "For each example, the supplied SAS is the reference property "
            "value for that example. Do not copy an example unnecessarily. "
            "Return only the generated SMILES. Do not include explanations, "
            "Markdown fences, labels, or multiple candidates."
        )
    elif names == {"tpsa", "sas"}:
        system_prompt = (
            "You are a molecular generation model. "
            "Generate exactly ONE valid molecular SMILES string whose "
            "topological polar surface area (TPSA), calculated using "
            "RDKit rdMolDescriptors.CalcTPSA, and synthetic accessibility "
            "score (SAS), calculated using RDKit sascorer.calculateScore, "
            "are as close as possible to their requested targets. "
            "A valid SMILES is not sufficient by itself: try to match "
            "both requested properties simultaneously while producing "
            "a chemically plausible molecule. Treat the examples as "
            "demonstrations of the requested input-output format. "
            "For each example, the supplied TPSA and SAS values are the "
            "reference property values for that example. Do not copy an "
            "example unnecessarily. Return only the generated SMILES. "
            "Do not include explanations, Markdown fences, labels, "
            "or multiple candidates."
        )
    elif names == {"logp", "tpsa"}:
        system_prompt = (
            "You are a molecular generation model. "
            "Generate exactly ONE valid molecular SMILES string whose "
            "calculated octanol/water partition coefficient (LogP), "
            "calculated using RDKit Crippen MolLogP, and topological "
            "polar surface area (TPSA), calculated using RDKit "
            "rdMolDescriptors.CalcTPSA, are as close as possible to "
            "their requested targets. A valid SMILES is not sufficient "
            "by itself: try to match both requested properties "
            "simultaneously while producing a chemically plausible "
            "molecule. Treat the examples as demonstrations of the "
            "requested input-output format. For each example, the "
            "supplied LogP and TPSA values are the reference property "
            "values for that example. Do not copy an example "
            "unnecessarily. Return only the generated SMILES. "
            "Do not include explanations, Markdown fences, labels, "
            "or multiple candidates."
        )
    elif names == {"sas", "logp"}:
        system_prompt = (
            "You are a molecular generation model. "
            "Generate exactly ONE valid molecular SMILES string whose "
            "synthetic accessibility score (SAS), calculated using "
            "RDKit sascorer.calculateScore, and calculated octanol/water "
            "partition coefficient (LogP), calculated using RDKit "
            "Crippen MolLogP, are as close as possible to their "
            "requested targets. A valid SMILES is not sufficient by "
            "itself: try to match both requested properties "
            "simultaneously while producing a chemically plausible "
            "molecule. Treat the examples as demonstrations of the "
            "requested input-output format. For each example, the "
            "supplied SAS and LogP values are the reference property "
            "values for that example. Do not copy an example "
            "unnecessarily. Return only the generated SMILES. "
            "Do not include explanations, Markdown fences, labels, "
            "or multiple candidates."
        )
    elif names == {"logp", "sas", "tpsa"}:
        system_prompt = (
            "You are a molecular generation model. "
            "Generate exactly ONE valid molecular SMILES string whose "
            "calculated octanol/water partition coefficient (LogP), "
            "calculated using RDKit Crippen MolLogP, topological polar "
            "surface area (TPSA), calculated using RDKit "
            "rdMolDescriptors.CalcTPSA, and synthetic accessibility "
            "score (SAS), calculated using RDKit sascorer.calculateScore, "
            "are as close as possible to their requested targets. "
            "A valid SMILES is not sufficient by itself: try to match "
            "all three requested properties simultaneously while "
            "producing a chemically plausible molecule. Treat the "
            "examples as demonstrations of the requested input-output "
            "format. For each example, the supplied LogP, SAS, and TPSA "
            "values are the reference property values for that example. "
            "Do not copy an example unnecessarily. Return only the "
            "generated SMILES. Do not include explanations, Markdown "
            "fences, labels, or multiple candidates."
        )
    else:
        system_prompt = (
            "You are a computational chemistry assistant. "
            "Follow the requested output format exactly."
        )

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        return text, False

    return prompt, True

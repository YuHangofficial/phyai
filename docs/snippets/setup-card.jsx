export const SetupCard = ({ rows = {} }) => {
    const entries = Object.entries(rows)
    const [selections, setSelections] = useState(() =>
        Object.fromEntries(entries.map(([key]) => [key, 0]))
    )

    const isTuple = (option) => Array.isArray(option)
    const labelOf = (option) => (isTuple(option) ? option[0] : option)

    let command = ""
    for (const [key, options] of entries) {
        if (options.length > 0 && isTuple(options[0])) {
            const index = selections[key] ?? 0
            command = options[index]?.[1] || ""
            break
        }
    }

    const setSelection = (key, index) => {
        setSelections((previous) => ({ ...previous, [key]: index }))
    }

    return (
        <div className="phyai-setup-card not-prose">
            {entries.map(([key, options]) => {
                const selectedIndex = selections[key] ?? 0
                const interactive = options.length > 1
                return (
                    <div key={key} className="phyai-setup-card__row">
                        <div className="phyai-setup-card__label">{key}</div>
                        <div className="phyai-setup-card__options" role="group" aria-label={key}>
                            {options.map((option, index) => {
                                const label = labelOf(option)
                                const selected = selectedIndex === index
                                return (
                                    <button
                                        key={label}
                                        type="button"
                                        disabled={!interactive}
                                        aria-pressed={interactive ? selected : undefined}
                                        data-selected={selected ? "true" : "false"}
                                        data-interactive={interactive ? "true" : "false"}
                                        onClick={() => setSelection(key, index)}
                                        className="phyai-setup-card__option"
                                    >
                                        {label}
                                    </button>
                                )
                            })}
                        </div>
                    </div>
                )
            })}
            <div className="phyai-setup-card__row phyai-setup-card__command-row">
                <div className="phyai-setup-card__label">Run this command</div>
                <div className="phyai-setup-card__command-wrap">
                    <pre className="phyai-setup-card__command" aria-live="polite">
                        {command || "—"}
                    </pre>
                </div>
            </div>
        </div>
    )
}

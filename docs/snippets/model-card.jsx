export const ModelCard = ({ title, subtitle, icon, rows = {} }) => {
    const entries = Object.entries(rows)

    const renderValue = (value) => {
        if (value === null || value === undefined) {
            return <span className="phyai-model-card__empty">&mdash;</span>
        }
        if (Array.isArray(value)) {
            return (
                <div className="phyai-model-card__tags">
                    {value.map((tag, index) => (
                        <span key={index} className="phyai-model-card__tag">
                            {tag}
                        </span>
                    ))}
                </div>
            )
        }
        if (typeof value === "string" || typeof value === "number") {
            return <span className="phyai-model-card__text">{value}</span>
        }
        return value
    }

    const hasHeader = title || subtitle || icon

    return (
        <div className="phyai-model-card not-prose">
            {hasHeader && (
                <div className="phyai-model-card__header">
                    {icon && <div className="phyai-model-card__icon">{icon}</div>}
                    <div className="phyai-model-card__heading">
                        {title && <div className="phyai-model-card__title">{title}</div>}
                        {subtitle && <div className="phyai-model-card__subtitle">{subtitle}</div>}
                    </div>
                </div>
            )}

            <div className="phyai-model-card__rows">
                {entries.map(([key, value]) => (
                    <div key={key} className="phyai-model-card__row">
                        <div className="phyai-model-card__label">{key}</div>
                        <div className="phyai-model-card__value">{renderValue(value)}</div>
                    </div>
                ))}
            </div>
        </div>
    )
}

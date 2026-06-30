// Small set of decorative line-icons matched against common ingredient
// keywords. Falls back to the CSS ✦ glyph (via the default li::before
// rule) when nothing matches, so this is additive — never required.
// Order matters: more specific compound terms (buttermilk, cream cheese)
// are checked before their broader substrings (butter, milk/cream) so
// they don't get misclassified by an earlier, looser pattern.
const INGREDIENT_ICONS = [
  { match: /\bbuttermilk/i,                  icon: "🥛" },
  { match: /\bcream cheese/i,                icon: "🧀" },
  { match: /\begg/i,                         icon: "🥚" },
  { match: /\bflour/i,                       icon: "🌾" },
  { match: /\b(butter|margarine)/i,          icon: "🧈" },
  { match: /\b(sugar|caster)/i,              icon: "🍬" },
  { match: /\b(milk|cream)/i,                icon: "🥛" },
  { match: /\b(chocolate|cocoa|cacao)/i,     icon: "🍫" },
  { match: /\bvanilla/i,                     icon: "🌿" },
  { match: /\bsalt/i,                        icon: "🧂" },
  { match: /\b(baking soda|baking powder)/i, icon: "✨" },
  { match: /\boil/i,                         icon: "🫗" },
  { match: /\b(cinnamon|spice|nutmeg)/i,     icon: "🌰" },
  { match: /\b(strawberr|berry|fruit)/i,     icon: "🍓" },
  { match: /\bcheese/i,                      icon: "🧀" },
  { match: /\bwater/i,                       icon: "💧" },
];

function getIngredientIcon(text) {
  const found = INGREDIENT_ICONS.find((entry) => entry.match.test(text));
  return found ? found.icon : null;
}

function MessageBubble({
  role, text, recipe, imageUrls = [],
  fileUrl, fileName, isFile, isImage, detectedCakeType,
}) {

  if (isFile) {
    if (isImage && fileUrl) {
      return (
        <div className={`message ${role}`}>
          <img
            src={fileUrl} alt={fileName}
            style={{
              maxWidth: "220px", maxHeight: "220px",
              objectFit: "cover", borderRadius: "10px",
              display: "block", marginBottom: text ? "6px" : 0, cursor: "pointer",
            }}
            onClick={() => window.open(fileUrl, "_blank")}
          />
          {text && text !== "📷 Image" && <p>{text}</p>}
        </div>
      );
    }
    return (
      <div className={`message ${role}`}>
        <a href={fileUrl} target="_blank" rel="noreferrer">{fileName}</a>
      </div>
    );
  }

  // Structured recipe-card template — ONLY rendered when a fresh RAG
  // retrieval produced a structured `recipe` object. Off-topic deflections
  // and follow-up answers never set `recipe`, so they fall through to the
  // plain text bubble below automatically. The "recipe-card" class is an
  // explicit styling hook (see chat.css) alongside the :has(h2) selector,
  // so the gold foil treatment doesn't depend solely on newer CSS support.
  if (recipe) {
    return (
      <div className={`message ${role} recipe-card`}>

        {/* Vision detection badge */}
        {detectedCakeType && (
          <p style={{ fontSize: "0.8em", opacity: 0.65, marginBottom: "6px" }}>
            🎂 Detected: <strong>{detectedCakeType}</strong>
          </p>
        )}

        {/* Scaling badge */}
        {recipe.scaled && recipe.scale_info && (
          <div style={{
            display: "inline-block",
            background: "linear-gradient(135deg, #D4A94A, #E3C271)",
            color: "#160B07",
            borderRadius: "20px",
            padding: "3px 12px",
            fontSize: "11px",
            fontWeight: 700,
            marginBottom: "10px",
            letterSpacing: "0.3px",
          }}>
            ⚖️ {recipe.scale_info}
          </div>
        )}

        <h2>{recipe.recipe_name}</h2>

        <p><strong>Prep Time:</strong> {recipe.prep_time}</p>
        <p><strong>Servings:</strong> {recipe.servings}</p>

        <h3>Ingredients</h3>
        <ul>
          {recipe.ingredients?.map((item, i) => {
            const icon = getIngredientIcon(item);
            return (
              <li key={i} className={icon ? "has-icon" : undefined}>
                {icon && <span style={{ marginRight: "2px" }}>{icon}</span>}
                {item}
              </li>
            );
          })}
        </ul>

        <h3>Instructions</h3>
        <ol>
          {recipe.instructions?.map((step, i) => <li key={i}>{step}</li>)}
        </ol>

        {recipe.notes && (
          <>
            <h3>Notes</h3>
            <p>{recipe.notes}</p>
          </>
        )}

        {/* Horizontal scrollable image strip */}
        {imageUrls.length > 0 && (
          <div style={{
            display: "flex", flexDirection: "row",
            gap: "10px", overflowX: "auto",
            paddingBottom: "8px", marginTop: "14px",
            scrollbarWidth: "thin",
          }}>
            {imageUrls.map((img, i) => (
              <img
                key={i} src={img} alt={`recipe-${i + 1}`}
                style={{
                  width: "160px", height: "120px",
                  objectFit: "cover", borderRadius: "8px",
                  flexShrink: 0, cursor: "pointer",
                  border: "1.5px solid #4F2F1C",
                }}
                onClick={() => window.open(img, "_blank")}
              />
            ))}
          </div>
        )}
      </div>
    );
  }

  // Plain text bubble — used for off-topic deflections, follow-up answers,
  // and "no matching recipe found" responses.
  return (
    <div className={`message ${role}`}>
      {detectedCakeType && (
        <p style={{ fontSize: "0.8em", opacity: 0.65, marginBottom: "4px" }}>
          🎂 Detected: <strong>{detectedCakeType}</strong>
        </p>
      )}
      {text}
    </div>
  );
}

export default MessageBubble;

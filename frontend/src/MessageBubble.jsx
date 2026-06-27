function MessageBubble({
  role,
  text,
  recipe,
  imageUrls = [],
  fileUrl,
  fileName,
  isFile,
  isImage,
  detectedCakeType,
}) {

  if (isFile) {
    if (isImage && fileUrl) {
      return (
        <div className={`message ${role}`}>
          <img
            src={fileUrl}
            alt={fileName}
            style={{
              maxWidth: "220px",
              maxHeight: "220px",
              objectFit: "cover",
              borderRadius: "10px",
              display: "block",
              marginBottom: text ? "6px" : 0,
              cursor: "pointer",
            }}
            onClick={() => window.open(fileUrl, "_blank")}
          />
          {text && text !== "📷 Image" && <p>{text}</p>}
        </div>
      );
    }

    return (
      <div className={`message ${role}`}>
        <a href={fileUrl} target="_blank" rel="noreferrer">
          {fileName}
        </a>
      </div>
    );
  }

  // FIX: recipe block now uses imageUrls (plural) which matches
  // both the botMsg shape in chat.jsx and the API response key.
  if (recipe) {
    return (
      <div className={`message ${role}`}>
        {detectedCakeType && (
          <p style={{ fontSize: "0.85em", opacity: 0.7, marginBottom: "4px" }}>
            🎂 Detected: {detectedCakeType}
          </p>
        )}

        <h2>{recipe.recipe_name}</h2>

        <p>
          <strong>Prep Time:</strong> {recipe.prep_time}
        </p>

        <p>
          <strong>Servings:</strong> {recipe.servings}
        </p>

        <h3>Ingredients</h3>
        <ul>
          {recipe.ingredients?.map((item, index) => (
            <li key={index}>{item}</li>
          ))}
        </ul>

        <h3>Instructions</h3>
        <ol>
          {recipe.instructions?.map((step, index) => (
            <li key={index}>{step}</li>
          ))}
        </ol>

        {recipe.notes && (
          <>
            <h3>Notes</h3>
            <p>{recipe.notes}</p>
          </>
        )}

        {imageUrls.length > 0 && (
          <div
            style={{
              display: "flex",
              flexDirection: "row",
              gap: "10px",
              overflowX: "auto",
              paddingBottom: "8px",
              marginTop: "12px",
              scrollbarWidth: "thin",
            }}
          >
            {imageUrls.map((img, index) => (
              <img
                key={index}
                src={img}
                alt={`recipe-${index + 1}`}
                style={{
                  width: "160px",
                  height: "120px",
                  objectFit: "cover",
                  borderRadius: "8px",
                  flexShrink: 0,
                  cursor: "pointer",
                }}
                onClick={() => window.open(img, "_blank")}
              />
            ))}
          </div>
        )}
      </div>
    );
  }

  // Fallback: plain text bubble (errors, no-match responses)
  return (
    <div className={`message ${role}`}>
      {detectedCakeType && (
        <p style={{ fontSize: "0.85em", opacity: 0.7, marginBottom: "4px" }}>
          🎂 Detected: {detectedCakeType}
        </p>
      )}
      {text}
    </div>
  );
}

export default MessageBubble;

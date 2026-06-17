function MessageBubble({
  role,
  text,
  recipe,
  imageUrls = [],
  fileUrl,
  fileName,
  isFile,
}) {

  if (isFile) {
    return (
      <div className={`message ${role}`}>
        <a
          href={fileUrl}
          target="_blank"
          rel="noreferrer"
        >
          {fileName}
        </a>
      </div>
    );
  }

  if (recipe) {
    return (
      <div className={`message ${role}`}>

        <h2>{recipe.recipe_name}</h2>

        <p>
          <strong>Prep Time:</strong>{" "}
          {recipe.prep_time}
        </p>

        <p>
          <strong>Servings:</strong>{" "}
          {recipe.servings}
        </p>

        <h3>Ingredients</h3>

        <ul>
          {recipe.ingredients?.map((item, index) => (
            <li key={index}>
              {item}
            </li>
          ))}
        </ul>

        <h3>Instructions</h3>

        <ol>
          {recipe.instructions?.map((step, index) => (
            <li key={index}>
              {step}
            </li>
          ))}
        </ol>

        {recipe.notes && (
          <>
            <h3>Notes</h3>
            <p>{recipe.notes}</p>
          </>
        )}

        {imageUrls.map((img, index) => (
          <img
            key={index}
            src={img}
            alt="recipe"
            className="recipe-image"
          />
        ))}

      </div>
    );
  }

  return (
    <div className={`message ${role}`}>
      {text}
    </div>
  );
}

export default MessageBubble;
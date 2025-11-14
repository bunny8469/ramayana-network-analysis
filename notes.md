- [ ] centrality-based main characters. interactions with main characters have more weight
- [ ] interactions distributed over time, will have more weight (for example, the same number of interactions within a larger fram of time is important than that of a small frame of time)
- [ ] sentiment spectrum (themes)
- [ ] main characters & their relation evolution over time
- [ ] character clustering / core analysis

- region??

Problems
- Database contains pronouns (he, she, him, her). So have to replace that with proper contextual names (Disambiguation)
- Rama = Raghava? 
- should also replace "his father" -> "Dasaratha" 

- CCD Analysis
- Sentiment Analysis (green-red graph)

## Dataset pre-processing
1. finding dataset -- initial dataset was poetic, found a sanskrit -> english explanation mapped dataset which had narrative capacity
2. pronoun resolution / disambiguition -- replacing he / him with character names according to context
2.5. "his brother" / "his father" -> actual names according to context

3. extracting proper nouns from the text (character names, places, entities, ..)
4. character names and aliases mapping. mapping rama -> raghava, katshulaya. + similar spelling mapping. lakshman -> lakshmana, lakshmanan, ...| validated mapping after that

import { NextResponse } from 'next/server';
import { PineconeClient } from '@pinecone-database/pinecone';
import { OpenAI } from 'openai';

const systemPrompt = `
You are a "Rate My Professor" assistant designed to help students find the best professors based on their queries. Your task is to provide the top 3 professors that match the student's request, using a combination of retrieval and generative techniques.

Instructions:

Understand the Query:

Analyze the student’s question to determine their specific needs or preferences. This could include factors like subject area, teaching style, difficulty level, or personal preferences.
Retrieve Relevant Information:

Use a database of professor ratings, reviews, and subject expertise to retrieve a list of potential professors who match the criteria specified by the student.
Generate Top Recommendations:

From the retrieved information, identify and rank the top 3 professors who best match the query. Provide details such as their name, department, a brief summary of their teaching style or strengths, and any relevant ratings or reviews.
Format Your Response:

Present the information clearly and concisely. Include the professor's name, department, a brief description of why they are recommended, and any notable ratings or reviews.
Example:

Student Query: "I’m looking for a professor who is great at teaching organic chemistry and has a reputation for being very approachable."
Response:
Dr. Jane Smith - Department of Chemistry
Summary: Known for her engaging lectures and approachable demeanor, Dr. Smith receives high praise for her clear explanations and willingness to help students outside of class.
Rating: 4.7/5
Dr. Michael Lee - Department of Chemistry
Summary: Dr. Lee is celebrated for his interactive teaching methods and supportive attitude. Students appreciate his practical approach to complex concepts.
Rating: 4.6/5
Dr. Emily Davis - Department of Chemistry
Summary: With a strong focus on student engagement and accessibility, Dr. Davis is known for her thorough understanding of organic chemistry and dedication to student success.
Rating: 4.5/5
Additional Notes:

Ensure that the recommendations are based on the most recent and relevant data available.
Maintain an objective tone and provide information that can help students make informed decisions.
`;

export async function POST(req) {
    try {
        const data = await req.json();

        // Check if the environment variables are set
        if (!process.env.PINECONE_API_KEY) {
            throw new Error('PINECONE_API_KEY is not set');
        }

        const pineconeClient = new PineconeClient();
        await pineconeClient.init({
            apiKey: process.env.PINECONE_API_KEY,
        });

        const index = pineconeClient.Index('rag').namespace('ns1');
        const openai = new OpenAI();

        const text = data[data.length - 1].content;
        const embeddingResponse = await openai.embeddings.create({
            model: 'text-embedding-ada-003',
            input: text,
        });

        const embedding = embeddingResponse.data[0].embedding;

        const results = await index.query({
            topK: 3,
            includeMetadata: true,
            vector: embedding,
        });

        let resultString = '';
        results.matches.forEach((match) => {
            resultString += `
            Professor: ${match.id}
            Review: ${match.metadata.review || 'No review available'}
            Subject: ${match.metadata.subject || 'No subject available'}
            Stars: ${match.metadata.stars || 'No stars available'}
            \n\n`;
        });

        const lastMessage = data[data.length - 1];
        const lastMessageContent = lastMessage.content + resultString;
        const lastDataWithoutLastMessage = data.slice(0, data.length - 1);

        const completionStream = await openai.chat.completions.create({
            messages: [
                { role: 'system', content: systemPrompt },
                ...lastDataWithoutLastMessage,
                { role: 'user', content: lastMessageContent },
            ],
            model: 'gpt-4',
            stream: true,
        });

        const stream = new ReadableStream({
            async start(controller) {
                try {
                    for await (const chunk of completionStream) {
                        const content = chunk.choices[0]?.delta?.content;
                        if (content) {
                            controller.enqueue(new TextEncoder().encode(content));
                        }
                    }
                } catch (err) {
                    controller.error(err);
                } finally {
                    controller.close();
                }
            },
        });

        return new NextResponse(stream);
    } catch (error) {
        console.error('Error processing request:', error);
        return new NextResponse('Internal Server Error', { status: 500 });
    }
}

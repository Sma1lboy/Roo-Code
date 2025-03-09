import * as vscode from "vscode"
import { SingleCompletionHandler } from ".."
import { BaseProvider } from "./base-provider"
import { ApiHandlerOptions, ModelInfo } from "../../shared/api"
import { ApiStreamChunk } from "../transform/stream"
import OpenAI from "openai"
import Anthropic from "@anthropic-ai/sdk/index.mjs"
import { convertToOpenAiMessages } from "../transform/openai-format"
export interface TabbyConfig {
	endpoint: string
	apiKey?: any
}

const TABBY_EXTENSION_ID = "TabbyML.vscode-tabby"

export class TabbyHandler extends BaseProvider implements SingleCompletionHandler {
	protected options: ApiHandlerOptions
	private endpoint: string
	private token: string
	private client: OpenAI | null = null
	private initialized: boolean = false
	private initPromise: Promise<void> | null = null

	constructor(options: ApiHandlerOptions) {
		super()
		this.options = options
		this.endpoint = options.tabbyBaseUrl || ""
		this.token = options.tabbyApiKey || ""
	}

	private async ensureInitialized() {
		if (this.initialized) return

		if (!this.initPromise) {
			this.initPromise = this.initialize()
		}

		try {
			await this.initPromise
			this.initialized = true
		} catch (error) {
			this.initPromise = null
			throw error
		}
	}

	// Returns model configuration used for completions.
	getModel(): { id: string; info: ModelInfo } {
		return {
			id: this.options.tabbyModelId || "",
			info: {
				maxTokens: -1,
				contextWindow: 128_000,
				supportsImages: false,
				supportsPromptCache: false,
				inputPrice: 0,
				outputPrice: 0,
			},
		}
	}

	// Creates a message stream for completion.
	async *createMessage(
		systemPrompt: string,
		messages: Anthropic.Messages.MessageParam[],
	): AsyncGenerator<ApiStreamChunk> {
		try {
			await this.ensureInitialized()

			if (!this.client) {
				throw new Error("OpenAI client is not initialized")
			}

			const systemMessage: OpenAI.Chat.ChatCompletionSystemMessageParam = {
				role: "system",
				content: systemPrompt,
			}

			const convertedMessages = convertToOpenAiMessages(messages)
			const openAiMessages = [systemMessage, ...convertedMessages]

			try {
				const response = await this.client.chat.completions.create({
					model: this.options.tabbyModelId || "",
					messages: openAiMessages,
					max_tokens: 1024,
					temperature: 0.6,
					stream: true,
				})

				let fullText = ""
				for await (const chunk of response) {
					const content = chunk.choices[0]?.delta?.content || ""
					if (content) {
						fullText += content
						yield { type: "text", text: content }
					}
				}

				const inputText = openAiMessages.map((m) => m.content).join(" ")
				const estimatedInputTokens = Math.ceil(inputText.length / 4)
				const estimatedOutputTokens = Math.ceil(fullText.length / 4)

				yield {
					type: "usage",
					inputTokens: estimatedInputTokens,
					outputTokens: estimatedOutputTokens,
				}
			} catch (error) {
				const result = await this.completePrompt(
					openAiMessages.map((m) => `${m.role}: ${m.content}`).join("\n"),
				)

				const chunks = result.split(/(?<=\. |\n)/)
				for (const chunk of chunks) {
					if (chunk.trim()) {
						yield { type: "text", text: chunk }
					}
				}

				const inputText = openAiMessages.map((m) => m.content).join(" ")
				const estimatedInputTokens = Math.ceil(inputText.length / 4)
				const estimatedOutputTokens = Math.ceil(result.length / 4)

				yield {
					type: "usage",
					inputTokens: estimatedInputTokens,
					outputTokens: estimatedOutputTokens,
				}
			}
		} catch (error) {
			throw error
		}
	}
	// Initializes the handler with the provided endpoint and token
	private async initialize() {
		try {
			if (!this.endpoint) {
				throw new Error("Tabby endpoint is not configured")
			}

			// Create the OpenAI client using the endpoint and token
			this.client = new OpenAI({
				baseURL: `${this.endpoint}/v1`,
				apiKey: this.token || "dummy", // Use dummy if no token provided
				defaultHeaders: { "Content-Type": "application/json" },
			})
		} catch (error) {
			const errorMessage = `Tabby initialization failed: ${error}`
			throw error
		}
	}

	// Uses the stored OpenAI client configured with the Tabby endpoint to complete the prompt.
	async completePrompt(prompt: string): Promise<string> {
		try {
			await this.ensureInitialized()

			if (!this.client) {
				throw new Error("OpenAI client is not initialized")
			}

			const response = await this.client.chat.completions.create({
				model: this.options.tabbyModelId || "",
				messages: [{ role: "user", content: prompt }],
				max_tokens: 1024,
				temperature: 1,
				stream: true,
			})
			const chunks: string[] = []
			for await (const chunk of response) {
				chunks.push(chunk.choices[0]?.delta.content || "")
			}
			const resultText = chunks.join("")
			return resultText
		} catch (error) {
			const errorMessage = `Error in completePrompt: ${error instanceof Error ? error.message : "Unknown error"}`
			throw error
		}
	}
}

export async function initTabbyConfig(): Promise<TabbyConfig> {
	try {
		const config = vscode.workspace.getConfiguration("tabby")
		const endpoint = config.get("endpoint", "")

		const tabbyExtension = vscode.extensions.getExtension(TABBY_EXTENSION_ID)

		if (!tabbyExtension) {
			return { endpoint }
		}

		let tabbyApi
		if (tabbyExtension.isActive) {
			tabbyApi = tabbyExtension.exports
		} else {
			tabbyApi = await tabbyExtension.activate()
		}

		if (tabbyApi && typeof tabbyApi.canGetToken === "function") {
			const token = (await tabbyApi.canGetToken("rooveterinaryinc.roo - cline")).token
			return {
				endpoint,
				apiKey: token,
			}
		}
		return {
			endpoint,
		}
	} catch (error) {
		throw error
	}
}

export async function getTabbyModels(
	tabbyBaseUrl: string = "http://localhost:8080",
	tabbyApiKey?: string,
): Promise<string[]> {
	if (!tabbyBaseUrl || tabbyApiKey === undefined) {
		return []
	}
	const response = await (
		await fetch(`${tabbyBaseUrl}/v1beta/models`, {
			headers: {
				Authorization: `Bearer ${tabbyApiKey}`,
			},
		})
	).json()
	const models = response.chat
	return models || []
}

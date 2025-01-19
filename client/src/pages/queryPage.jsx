import React, { useState } from 'react'
import { Card, CardContent } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "../../components/ui/scroll-area"

const QueryPage = () => {
    const [messages, setMessages] = useState([])
    const [input, setInput] = useState('')
    const [isLoading, setIsLoading] = useState(false)


    const handleSubmit = async (e) => {
        e.preventDefault()
        if (!input.trim()) return
        setIsLoading(true)

        // Add user message
        const userMessage = { role: 'user', content: input }
        setMessages(prev => [...prev, userMessage])
        setInput('')

        try {
            const response = await fetch('http://localhost:8000/query-rag', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question: input }),
            })
            const data = await response.json()

            // Add bot response
            const botMessage = { role: 'assistant', content: data }
            setMessages(prev => [...prev, botMessage])
        } catch (error) {
            console.error('Error:', error)
        } finally {
            setIsLoading(false)

        }
    }

    return (
        <div className="flex flex-col h-screen max-w-2xl mx-auto p-4">
            <Card className="flex-1 mb-4 max-h-screen">
                <CardContent className="p-4 h-full">
                    <ScrollArea className="h-full">
                        {messages.map((message, index) => (
                            <div
                                key={index}
                                className={`mb-4 flex ${message.role === 'user'
                                    ? 'justify-end'
                                    : 'justify-start'
                                    }`}
                            >
                                <div
                                    className={`max-w-[80%] p-4 rounded-lg ${message.role === 'user'
                                        ? 'bg-blue-600 text-white rounded-br-none'
                                        : 'bg-slate-400 text-secondary-foreground rounded-bl-none'
                                        }`}
                                >
                                    {message.content}
                                </div>
                            </div>
                        ))}
                    </ScrollArea>
                </CardContent>
            </Card>

            <form onSubmit={handleSubmit} className="flex gap-2">
                <Input
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Type your message..."
                    className="flex-1"
                />
                <Button type="submit" disabled={isLoading}>
                    {isLoading ? "Sending..." : "Send"}
                </Button>
            </form>
        </div>
    )
}

export default QueryPage
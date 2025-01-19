import React, { useState, useEffect } from 'react'
import 'react-big-calendar/lib/css/react-big-calendar.css'

import { momentLocalizer, Calendar } from 'react-big-calendar'
import moment from 'moment'

import { NavBar } from '../components/navbar'

const localizer = momentLocalizer(moment)

const CalendarPage = () => {
    const [events, setEvents] = useState([])

    useEffect(() => {
        // Fetch Todo data from backend
        const fetchEvents = async () => {
            try {
                const response = await fetch('http://localhost:8000/get-todos')
                const data = await response.json()
                // Transform the data into the required format
                const transformedEvents = data.map(item => ({
                    title: item.task,
                    start: new Date(item.deadline),
                    end: new Date(item.deadline)
                }))

                setEvents(transformedEvents);
            } catch (error) {
                console.error('Failed to fetch events:', error)
                setEvents([])
            }
        }
        fetchEvents()


    }, [])
    return (
        <div className="max-w-2xl mx-auto p-4">
            <Calendar
                localizer={localizer}
                events={events}
                startAccessor="start"
                endAccessor="end"
                showMultiDayTimes
                style={{ height: 500 }}
            />
        </div>
    )
}

export default CalendarPage
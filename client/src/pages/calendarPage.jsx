import React, { useState, useEffect } from 'react'
import 'react-big-calendar/lib/css/react-big-calendar.css'

import { momentLocalizer, Calendar } from 'react-big-calendar'
import moment from 'moment'

import { NavBar } from '../components/navbar'

const localizer = momentLocalizer(moment)

const CalendarPage = () => {
    const [events, setEvents] = useState([])

    useEffect(() => {
        // Simulate fetching data from an API
        const fetchedEvents = [
            {
                id: 1,
                allDay: true,
                title: 'Long Event',
                start: new Date(2025, 1, 23),
                end: new Date(2025, 1, 23),
            },
            {
                id: 2,
                allDay: true,
                title: 'Another Event',
                start: new Date(2025, 1, 25),
                end: new Date(2025, 1, 26),
            },
        ]

        const fetchEvents = async () => {
            try {
                const response = await fetch('https://api.example.com/events')
                const data = await response.json()
                setEvents(data)
            } catch (error) {
                console.error('Failed to fetch events:', error)
                setEvents([])
            }
        }
        fetchEvents()


    }, [])
    console.log(events)
    return (
        <>
            <Calendar
                localizer={localizer}
                events={events}
                startAccessor="start"
                endAccessor="end"
                showMultiDayTimes
                style={{ height: 500 }}
            />
        </>
    )
}

export default CalendarPage
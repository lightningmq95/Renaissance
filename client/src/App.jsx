import { useState } from 'react'
import { BrowserRouter, Routes, Route } from "react-router-dom"
import CalendarPage from './pages/calendarPage'
import QueryPage from './pages/queryPage'
import Home from './pages/home'
import { NavBar } from './components/navbar'
import './App.css'

function App() {
  const [count, setCount] = useState(0)

  return (
    <>
      <BrowserRouter>
        <NavBar />
        <Routes>
          <Route path="/" >
            <Route index element={<Home />} />
            <Route path="calendar" element={<CalendarPage />} />
            <Route path='query' element={<QueryPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </>
  )
}

export default App

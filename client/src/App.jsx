import { useState } from 'react'
import { BrowserRouter, Routes, Route } from "react-router-dom"
import CalendarPage from './pages/calendarPage'
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
            {/* <Route path="blogs" element={<Blogs />} />
            <Route path="contact" element={<Contact />} />
            <Route path="*" element={<NoPage />} /> */}
          </Route>
        </Routes>
      </BrowserRouter>
    </>
  )
}

export default App

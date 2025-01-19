import { useState } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Home from "./pages/home";
import reactLogo from "./assets/react.svg";
import viteLogo from "/vite.svg";
import "./App.css";
import CalendarPage from "./pages/calendarPage";
import { NavBar } from "./components/navbar";
import RealTime from "./pages/RealTime";
import Meetings from "./pages/meetings";
import OnlineMeeting from "./pages/onlineMeeting";

function App() {
  const [count, setCount] = useState(0);

  return (
    <>
      <BrowserRouter>
        <NavBar />
        <Routes>
          <Route path="/">
            <Route index element={<Home />} />
            <Route path="calendar" element={<CalendarPage />} />
            <Route path="realTime" element={<RealTime />} />
            <Route path="meetings" element={<Meetings />} />
            <Route path="onlineMeeting" element={<OnlineMeeting />} />
            {/* <Route path="blogs" element={<Blogs />} />
            <Route path="contact" element={<Contact />} />
            <Route path="*" element={<NoPage />} /> */}
          </Route>
        </Routes>
      </BrowserRouter>
    </>
  );
}

export default App;

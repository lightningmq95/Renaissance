import React from "react";
import { Button } from "@/components/ui/button";
import { Video, Clock } from "lucide-react";
import { useNavigate } from "react-router-dom";
const Home = () => {
  const navigate = useNavigate();
  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="py-12">
          {/* Hero Section */}
          <div className="text-center">
            <h1 className="text-4xl font-bold text-gray-900 sm:text-5xl md:text-6xl">
              Welcome to Our Platform
            </h1>
            <p className="mt-3 max-w-md mx-auto text-base text-gray-500 sm:text-lg md:mt-5 md:text-xl md:max-w-3xl">
              Choose how you want to connect with our services
            </p>
          </div>

          {/* Button Section */}
          <div className="mt-10">
            <div className="flex flex-col space-y-4 sm:flex-row sm:space-y-0 sm:space-x-4 justify-center">
              <Button
                size="lg"
                className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-700"
                onClick={() => navigate("/realTime")}
              >
                <Clock className="h-5 w-5" />
                <span>Real Time</span>
              </Button>

              <Button
                size="lg"
                className="flex items-center space-x-2 bg-green-600 hover:bg-green-700"
                onClick={() => navigate("/onlineMeeting")}
              >
                <Video className="h-5 w-5" />
                <span>Online Meeting</span>
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;

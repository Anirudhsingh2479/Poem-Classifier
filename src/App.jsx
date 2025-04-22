// import { useState } from 'react'
import { useState, useEffect } from 'react';
import './App.css'
import './index.css'
function App() {
  const [inputText, setInputText] = useState("");
  const [prediction, setPrediction] = useState("");
  const [loading, setLoading] = useState(false);
  const [fade, setFade] = useState(true);
  const [bgImage, setBgImage] = useState("/images/default.webp");
  //background images
  const genreBackgrounds = {
    Environment: "/images/environment.webp",
    Affection: "/images/affection.webp",
    Death: "/images/death.webp",
    Music: "/images/music.webp",
    default: "/images/default.webp",
    Sorrow: "/images/sorrow.jpg"
  };

  useEffect(() => {
    if (!prediction) return;

    setFade(false); // fade out
    const timeout = setTimeout(() => {
      const newBg = genreBackgrounds[prediction] || genreBackgrounds.default;
      setBgImage(newBg);
      setFade(true); // fade in
    }, 500); // duration of fade out

    return () => clearTimeout(timeout);
  }, [prediction]);

  const handlePredict = async ()=>{
    if(!inputText) return;
    setLoading(true);
    try {
      const res = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ poem: inputText }),
      });
      const data = await res.json();
      setPrediction(data.genre);
      
      // setLoading(false);
    } catch (error) {
      console.error("Prediction error:", err);
      setPrediction("Something went wrong.");
    }
    setLoading(false);
  };
  // const bgImage = genreBackgrounds[prediction] || genreBackgrounds.default
  return (
    <>
    <div
    style={{
      backgroundImage: `url(${bgImage})`,
      backgroundSize: 'cover',
      backgroundPosition: 'center',
    }}
    className={`h-dvh bg-gray-500 w-screen absolute flex items-center justify-center overflow-hidden md:p-56 transition-opacity duration-700 ease-in-out ${fade ? 'opacity-100' : 'opacity-0'}`}>
    </div>
    <div 
      className='relative z-10 h-dvh w-screen flex items-center justify-center overflow-hidden md:p-56 bg-none'
    >
      <div
      style = {{
      // backgroundColor: klch(0.94 0.09 167.81);
      }}
      className="relative z-20 inset-0 shadow-2xl bg-[oklch(0.94_0.09_167.81)] grid grid-cols-1 grid-rows-4 rounded-3xl place-items-center px-4">
        <div>
        <h1 className=' text-7xl'> POEM CLASSIFIER</h1>
        </div>

        <div className="bg-white shadow-md p-6 rounded-xl w-full max-w-md row-span-2">

        <textarea
          type="text"
          placeholder="Enter your poem here..."
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          className="w-full p-4 border rounded mb-4 align-text-top focus:outline-none resize-none h-40"

          >
          </textarea>
        <button
          onClick={handlePredict}
          className="w-full bg-blue-600 text-white py-2 rounded hover:bg-blue-700"
        >
          {loading ? "Predicting..." : "Predict"}
        </button>
        {!prediction && (
          <div className="mt-4 text-center text-gray-500">
              No prediction available yet.
          </div>
        )}
        
        {prediction && (
          <div className="mt-4 text-center text-gray-700">
            <strong>Genre:</strong> {prediction}
          </div>
        )}
        </div>
      </div>
    </div>
    </>
  )
}

export default App

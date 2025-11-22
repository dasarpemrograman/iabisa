import Navbar from "../components/navbar";
import DashboardLayout from "../components/dashboard-layout";

export default async function App() {
  // Simulasi loading
  await new Promise((resolve) => setTimeout(resolve, 2000)); // Dikurangi jadi 2s agar tidak terlalu lama
  
  return (
    <div className="flex h-screen flex-col bg-gray-50 overflow-hidden">
       {/* Tambahkan Blob background untuk efek glow global */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-[500px] h-[500px] bg-emerald-200/20 rounded-full blur-[100px]" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[500px] h-[500px] bg-teal-200/20 rounded-full blur-[100px]" />
      </div>

      <div className="z-10 flex flex-col h-full">
        <Navbar />
        <DashboardLayout />
      </div>
    </div>
  );
}
require "fileutils"
require "json"
require "net/http"
require "pathname"
require "uri"

module MlxLm
  # Minimal pure-Ruby equivalent of huggingface_hub.snapshot_download.
  # Cache layout matches the Python client so caches are mutually reusable.
  module Hub
    DEFAULT_ENDPOINT = "https://huggingface.co".freeze
    MAX_REDIRECTS = 5
    DEFAULT_TIMEOUT = 300

    module_function

    # Download a model snapshot from Hugging Face Hub.
    #
    # @param repo_id [String] e.g. "mlx-community/Llama-3.2-1B-Instruct-4bit"
    # @param revision [String] branch, tag, or commit SHA (default "main")
    # @param allow_patterns [Array<String>, String, nil] glob patterns to include
    # @param token [String, nil] defaults to ENV["HF_TOKEN"]
    # @param cache_dir [String, Pathname, nil] defaults to HF_HUB_CACHE / HF_HOME/hub / ~/.cache/huggingface/hub
    # @param endpoint [String, nil] defaults to HF_ENDPOINT or huggingface.co
    # @return [Pathname] absolute path to the snapshot directory
    def snapshot_download(repo_id, revision: "main", allow_patterns: nil, token: nil, cache_dir: nil, endpoint: nil)
      endpoint ||= ENV["HF_ENDPOINT"] || DEFAULT_ENDPOINT
      token ||= ENV["HF_TOKEN"]
      cache_dir = resolve_cache_dir(cache_dir)
      patterns = normalize_patterns(allow_patterns)

      info = fetch_model_info(endpoint, repo_id, revision, token)
      sha = info.fetch("sha")
      siblings = info.fetch("siblings").map { |s| s.fetch("rfilename") }

      repo_folder = cache_dir.join("models--#{repo_id.gsub("/", "--")}")
      snapshot_dir = repo_folder.join("snapshots", sha)
      FileUtils.mkdir_p(snapshot_dir)
      FileUtils.mkdir_p(repo_folder.join("refs"))
      File.write(repo_folder.join("refs", revision), sha)

      siblings.each do |rel|
        next unless pattern_match?(rel, patterns)
        target = snapshot_dir.join(rel)
        next if target.file? && target.size > 0

        FileUtils.mkdir_p(target.dirname)
        url = "#{endpoint}/#{repo_id}/resolve/#{revision}/#{rel}"
        download_file(url, target.to_s, token)
      end

      snapshot_dir
    end

    def resolve_cache_dir(explicit)
      return Pathname.new(explicit) if explicit
      if (v = ENV["HF_HUB_CACHE"]) && !v.empty?
        Pathname.new(v)
      elsif (v = ENV["HF_HOME"]) && !v.empty?
        Pathname.new(v).join("hub")
      else
        Pathname.new(Dir.home).join(".cache", "huggingface", "hub")
      end
    end

    def normalize_patterns(p)
      return nil if p.nil?
      p.is_a?(::Array) ? p : [p]
    end

    def pattern_match?(filename, patterns)
      return true if patterns.nil? || patterns.empty?
      patterns.any? do |pat|
        File.fnmatch(pat, filename, File::FNM_PATHNAME) ||
          File.fnmatch(pat, File.basename(filename))
      end
    end

    def fetch_model_info(endpoint, repo_id, revision, token)
      url = "#{endpoint}/api/models/#{repo_id}/revision/#{revision}"
      body = http_get_body(url, token)
      JSON.parse(body)
    end

    def http_get_body(url, token, limit = MAX_REDIRECTS)
      raise "Too many redirects fetching #{url}" if limit <= 0
      uri = URI.parse(url)
      Net::HTTP.start(uri.host, uri.port, use_ssl: uri.scheme == "https") do |http|
        req = Net::HTTP::Get.new(uri.request_uri)
        req["Authorization"] = "Bearer #{token}" if token && uri.host.end_with?("huggingface.co")
        req["User-Agent"] = user_agent
        resp = http.request(req)
        case resp
        when Net::HTTPSuccess then resp.body
        when Net::HTTPRedirection then http_get_body(URI.join(url, resp["location"]).to_s, token, limit - 1)
        else raise "HTTP #{resp.code} #{resp.message} fetching #{url}: #{resp.body}"
        end
      end
    end

    def download_file(url, output_path, token = nil, limit = MAX_REDIRECTS)
      raise "Too many redirects fetching #{url}" if limit <= 0
      uri = URI.parse(url)
      Net::HTTP.start(uri.host, uri.port, use_ssl: uri.scheme == "https", read_timeout: DEFAULT_TIMEOUT) do |http|
        req = Net::HTTP::Get.new(uri.request_uri)
        req["Authorization"] = "Bearer #{token}" if token && uri.host.end_with?("huggingface.co")
        req["User-Agent"] = user_agent
        http.request(req) do |resp|
          case resp
          when Net::HTTPSuccess
            tmp = "#{output_path}.download.#{Process.pid}"
            File.open(tmp, "wb") { |f| resp.read_body { |chunk| f.write(chunk) } }
            FileUtils.mv(tmp, output_path)
          when Net::HTTPRedirection
            return download_file(URI.join(url, resp["location"]).to_s, output_path, token, limit - 1)
          else
            raise "HTTP #{resp.code} #{resp.message} fetching #{url}"
          end
        end
      end
    end

    def user_agent
      v = defined?(MlxLm::VERSION) ? MlxLm::VERSION : "unknown"
      "mlx-ruby-lm/#{v}"
    end

    private_class_method :resolve_cache_dir, :normalize_patterns, :pattern_match?,
      :fetch_model_info, :download_file, :user_agent
  end
end
